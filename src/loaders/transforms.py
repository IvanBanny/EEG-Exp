import numpy as np
import torch
from torch.utils.data import Dataset

class TransformWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform(sample[0]), sample[1]

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