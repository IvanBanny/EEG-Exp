import numpy as np
import torch
from torch.utils.data import Dataset

class TransformWrapper(Dataset):
    """Wrap a dataset with a per-sample transform pipeline.

    If `subject_normalizer` is set, the wrapped dataset is expected to
    yield 3-tuples (data, label, subject_id) and the normalizer is
    applied before the rest of the transform pipeline. Otherwise the
    dataset yields 2-tuples (data, label) and only the transform is
    applied.
    """

    def __init__(self, dataset, transform, subject_normalizer=None):
        self.dataset = dataset
        self.transform = transform
        self.subject_normalizer = subject_normalizer

    def __getitem__(self, idx):
        if self.subject_normalizer is not None:
            x, y, sid = self.dataset[idx]
            x = self.subject_normalizer(x, sid)
        else:
            sample = self.dataset[idx]
            x, y = sample[0], sample[1]
        return self.transform(x), y

    def __len__(self):
        return len(self.dataset)

class Compose:
    """Chain multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class ToTensor:
    """Convert numpy array to torch tensor, preserving or setting dtype."""
    def __init__(self, dtype=torch.bfloat16):
        self.dtype = dtype

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.dtype)
        return x.to(self.dtype)

class ZScoreNormalize:
    """Normalize across the entire spectrogram (all channels) to preserve relative amplitudes."""
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, x):
        # x shape: (channels, H, W)
        # Calculate mean/std across ALL dimensions
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + self.eps)


class PerSubjectZScore:
    """Stateful per-subject z-score normalization.

    Stats are fitted on a subject's training windows and then applied to
    every window from that subject, train or val. This mirrors the
    sklearn fit/transform split: the fit step needs dataset-level access
    and is owned by the factory, but the transform itself stays in the
    per-sample pipeline so inference is explicit.

    Stats are computed as scalar mean/std over all axes, matching
    `ZScoreNormalize` semantics (relative inter-channel amplitudes
    preserved). At deployment on a new subject, call `.fit` with a short
    calibration recording from that subject before inference.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.stats: dict[int, tuple[float, float]] = {}

    def fit(self, windows_by_subject) -> "PerSubjectZScore":
        """Compute and store (mean, std) per subject.

        Args:
            windows_by_subject: mapping subject_id -> array/tensor of
                shape (n_windows, *sample_shape) drawn from the
                training split only.
        """
        for sid, x in windows_by_subject.items():
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            t = t.to(torch.float32)
            self.stats[int(sid)] = (float(t.mean().item()),
                                    float(t.std().item()))
        return self

    def __call__(self, x: torch.Tensor, subject_id: int) -> torch.Tensor:
        sid = int(subject_id)
        stats = self.stats.get(sid)
        if stats is None:
            raise KeyError(
                f"PerSubjectZScore has no stats for subject {sid}. "
                f"Known subjects: {sorted(self.stats)}. "
                f"Call .fit({{sid: calibration_windows}}) first."
            )
        mean, std = stats
        return (x - mean) / (std + self.eps)

    def state_dict(self) -> dict:
        return {"eps": self.eps, "stats": dict(self.stats)}

    def load_state_dict(self, state: dict) -> "PerSubjectZScore":
        self.eps = state.get("eps", self.eps)
        self.stats = {int(k): (float(m), float(s))
                      for k, (m, s) in state["stats"].items()}
        return self


class PerChannelZScore:
    """Per-(subject, channel) z-score normalization.

    Stats are computed per channel over all training windows and
    timepoints of a subject. At call time the per-channel (mean, std)
    are broadcast across the time axis of a single window. This matches
    the EEGEncoder paper's `StandardScaler`-per-channel preprocessing
    while remaining usable inside our per-sample transform pipeline.

    Args:
        eps: Numerical floor for std.

    Stats schema:
        ``self.stats[subject_id]`` is a tuple ``(mean, std)`` of 1-D
        float32 tensors of length ``n_channels``.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def fit(self, windows_by_subject) -> "PerChannelZScore":
        """Compute per-channel (mean, std) tensors per subject.

        Args:
            windows_by_subject: mapping subject_id -> tensor / array of
                shape (n_windows, n_channels, *) from the training split.
        """
        for sid, x in windows_by_subject.items():
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            t = t.to(torch.float32)
            # Flatten across windows and time so each channel reduces to
            # a scalar (mean, std). Works for raw (n, C, T) and STFT
            # (n, C, F, T) layouts alike.
            flat = t.reshape(t.shape[0], t.shape[1], -1)
            mean = flat.mean(dim=(0, 2))
            std = flat.std(dim=(0, 2))
            self.stats[int(sid)] = (mean.clone(), std.clone())
        return self

    def __call__(self, x: torch.Tensor, subject_id: int) -> torch.Tensor:
        sid = int(subject_id)
        stats = self.stats.get(sid)
        if stats is None:
            raise KeyError(
                f"PerChannelZScore has no stats for subject {sid}. "
                f"Known subjects: {sorted(self.stats)}. "
                f"Call .fit({{sid: calibration_windows}}) first."
            )
        mean, std = stats
        # Broadcast (C,) -> (C, 1) for raw, (C, 1, 1) for STFT.
        shape = (-1,) + (1,) * (x.ndim - 1)
        mean = mean.to(x.device, x.dtype).view(*shape)
        std = std.to(x.device, x.dtype).view(*shape)
        return (x - mean) / (std + self.eps)

    def state_dict(self) -> dict:
        return {
            "eps": self.eps,
            "stats": {sid: (m.tolist(), s.tolist())
                      for sid, (m, s) in self.stats.items()},
        }

    def load_state_dict(self, state: dict) -> "PerChannelZScore":
        self.eps = state.get("eps", self.eps)
        self.stats = {
            int(sid): (torch.tensor(m, dtype=torch.float32),
                       torch.tensor(s, dtype=torch.float32))
            for sid, (m, s) in state["stats"].items()
        }
        return self


# vvv Might be problematic with bfloat16 which I'm (was) using vvv
class RobustScaler:
    """Median/IQR scaling - more robust to EEG artifacts and outliers."""
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, x):
        # x shape: (channels, time) or (channels, H, W)
        median = x.median(dim=-1, keepdim=True).values
        q75 = torch.quantile(x, 0.75, dim=-1, keepdim=True)
        q25 = torch.quantile(x, 0.25, dim=-1, keepdim=True)
        iqr = q75 - q25
        return (x - median) / (iqr + self.eps)
# ^^^ :skull:skull:skull ^^^

class LogCompress:
    """Apply log(1 + x) to compress STFT magnitude dynamic range."""

    def __call__(self, x):
        return torch.log1p(x)


class ClipOutliers:
    """Clip extreme values (common for EEG)."""
    def __init__(self, sigma=5.0):
        self.sigma = sigma

    def __call__(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        lower = mean - std * self.sigma
        upper = mean + std * self.sigma
        return torch.clamp(x, lower, upper)

class MinMaxNormalize:
    """Scale to [0, 1] range."""
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, x):
        min_val = x.min(dim=-1, keepdim=True).values
        max_val = x.max(dim=-1, keepdim=True).values
        return (x - min_val) / (max_val - min_val + self.eps)