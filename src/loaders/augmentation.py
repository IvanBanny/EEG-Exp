import torch

class GaussianNoise:
    """Add random noise."""
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std

class RandomScale:
    """Randomly scale amplitude."""
    def __init__(self, scale_fork=(0.8, 1.2)):
        self.scale_fork = scale_fork

    def __call__(self, x):
        scale = torch.empty(x.shape[0], 1, 1, dtype=x.dtype, device=x.device).uniform_(*self.scale_fork)
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
        mask = (torch.rand(x.shape[0], 1, 1, device=x.device) > self.p).to(x.dtype)
        return x * mask
