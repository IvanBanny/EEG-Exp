from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class GaussianNoise:
    """Add random noise."""
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std

class RandomScale:
    """Randomly scale amplitude per channel."""
    def __init__(self, scale_fork=(0.8, 1.2)):
        self.scale_fork = scale_fork

    def __call__(self, x):
        # Per-channel scale, broadcast over remaining dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        scale = torch.empty(shape, dtype=x.dtype, device=x.device).uniform_(*self.scale_fork)
        return x * scale

class TimeShift:
    """Random circular shift along the time axis."""
    def __init__(self, max_shift=16):
        self.max_shift = max_shift

    def __call__(self, x):
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)

class ChannelDropout:
    """Randomly zero out channels (simulates bad electrodes).

    Note: In a way - our current setup already does that with the two taped electrodes.
    One of them is still dead.
    """
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        # Per-channel mask, broadcast over remaining dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) > self.p).to(x.dtype)
        return x * mask


class IntraSubjectMixup(Dataset):
    """Within-(subject, class) mixup at the dataset level.

    For each item `i`, samples a partner `j` from the same (subject, class)
    bucket, draws lambda ~ Beta(alpha, alpha), and returns
        x_mixed = lambda * x_i + (1 - lambda) * x_j,
        y = y_i,
        [subject_id = subject_i]  if the underlying dataset emits one.

    The label is unchanged: within (subject, class) the two endpoints share the
    same target, so the mix is also valid for that target. This is a strict
    intra-class regulariser - it never crosses subjects or classes.

    Pairs are sampled at __getitem__ time using a numpy RandomState bound to
    `seed + worker_id` so dataloader workers diverge but each worker is
    deterministic. The wrapper is train-only; do not apply to val/test.

    Attributes:
        dataset: The wrapped EEGDataset / SubsetEEGDataset.
        alpha: Beta distribution parameter. 0 (or <=0) disables mixup.
        return_subject_id: Mirrors the wrapped dataset's setting so the
            transform pipeline still receives subject ids when needed.
    """

    def __init__(self, dataset: Dataset, alpha: float = 0.2,
                 seed: int = 12345):
        super().__init__()
        if alpha <= 0:
            raise ValueError("IntraSubjectMixup requires alpha > 0")
        self.dataset = dataset
        self.alpha = float(alpha)
        self._base_seed = int(seed)
        self._pools = self._build_pools(dataset)
        # Each worker derives its own RandomState in __getitem__ via worker_info
        self._fallback_rng = np.random.RandomState(self._base_seed)

    @staticmethod
    def _build_pools(dataset: Dataset) -> dict:
        """Build {(subject_id, class): np.ndarray of dataset indices}."""
        # SubsetEEGDataset and EEGDataset both expose `.labels` and metadata
        # via `.parent` in the subset case.
        from .torch_dataset import SubsetEEGDataset
        if isinstance(dataset, SubsetEEGDataset):
            parent = dataset.parent
            indices = np.asarray(dataset.indices)
            subjects = parent.metadata["subject"].to_numpy()[indices]
            labels = parent.labels[indices].numpy() \
                if hasattr(parent.labels, "numpy") else np.asarray(parent.labels)[indices]
        else:
            subjects = dataset.metadata["subject"].to_numpy()
            labels = dataset.labels.numpy() \
                if hasattr(dataset.labels, "numpy") else np.asarray(dataset.labels)

        pools = {}
        n = len(subjects)
        for i in range(n):
            key = (int(subjects[i]), int(labels[i]))
            pools.setdefault(key, []).append(i)
        return {k: np.asarray(v, dtype=np.int64) for k, v in pools.items()}

    @property
    def return_subject_id(self) -> bool:
        return getattr(self.dataset, "return_subject_id", False)

    @return_subject_id.setter
    def return_subject_id(self, value: bool) -> None:
        self.dataset.return_subject_id = value

    def _get_rng(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self._fallback_rng
        # One RNG per worker; cached on the wrapper inside the worker process
        cache = getattr(self, "_worker_rngs", None)
        if cache is None:
            cache = {}
            self._worker_rngs = cache
        rng = cache.get(info.id)
        if rng is None:
            rng = np.random.RandomState(self._base_seed + info.id)
            cache[info.id] = rng
        return rng

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if self.return_subject_id:
            x_i, y_i, sid = sample
        else:
            x_i, y_i = sample[0], sample[1]
            sid = None

        rng = self._get_rng()
        key = (int(sid) if sid is not None else None, int(y_i))
        if key[0] is None:
            # Wrapped dataset doesn't emit subject ids; fall back to (class) keying
            key = next(k for k in self._pools if k[1] == int(y_i))
        pool = self._pools.get(key)
        if pool is None or len(pool) <= 1:
            return sample  # no partner available; pass through

        # Pick a partner that is not self
        j = int(rng.choice(pool))
        if j == idx:
            # nudge by one in pool order
            pos = int(np.searchsorted(pool, idx))
            j = int(pool[(pos + 1) % len(pool)])

        partner = self.dataset[j]
        x_j = partner[0]
        lam = float(rng.beta(self.alpha, self.alpha))
        x_mix = lam * x_i + (1.0 - lam) * x_j

        if self.return_subject_id:
            return x_mix, y_i, sid
        return x_mix, y_i