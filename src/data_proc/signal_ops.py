"""Signal preprocessing operations for EEG data.

This module provides signal processing transforms (sliding windows, STFT), and utils
for surgically and quite violently injecting them into the MOABB array cache pipeline.
This is probably addressed in more detail in one of them Geneva Conventions.
It's a bit floppy, but it allows for caching widowed STFT arrays, which should be much faster.
It should also take a good amount of memory though due to windows, but I have enough don't worry.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import stft
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from moabb.datasets.bids_interface import StepType
from moabb.datasets.preprocessing import ForkPipelines


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window extraction.

    Single source of truth for all window-related parameters.
    Ensures consistency between data transform and label expansion.

    Attributes:
        sfreq: Sampling frequency in Hz.
        window_sec: Window length in seconds.
        window_overlap: Window overlap ratio [0.0 to 0.99].
        n_times: Number of time samples per trial.
    """

    sfreq: float
    window_sec: float
    window_overlap: float
    n_times: int

    @property
    def window_samples(self) -> int:
        """Window length in samples."""
        return int(self.window_sec * self.sfreq)

    @property
    def step_samples(self) -> int:
        """Step size between windows in samples."""
        return int(self.window_samples * (1 - self.window_overlap))

    @property
    def windows_per_trial(self) -> int:
        """Number of windows extracted from each trial."""
        return (self.n_times - self.window_samples) // self.step_samples + 1

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        if self.window_samples > self.n_times:
            raise ValueError(f"Window ({self.window_samples} samples) is larger than "
                             f"trial length ({self.n_times} samples).")
        if self.window_overlap < 0.0 or self.window_overlap > 0.99:
            raise ValueError(f"Bad window overlap value {self.window_overlap}, "
                             f"must be within interval [0.0, 0.99]")
        if self.windows_per_trial <= 0:
            raise ValueError(f"No windows fit in trial. window_samples={self.window_samples}, "
                             f"n_times={self.n_times}, step_samples={self.step_samples}.")


def make_sliding_window_fn(config: SlidingWindowConfig) -> Callable[[np.ndarray], np.ndarray]:
    """Create a sliding window transform function (no STFT).

    Args:
        config: Sliding window configuration.

    Returns:
        Transform function that takes (n_trials, n_channels, n_times) array
        and returns (n_windows_total, n_channels, window_samples) array,
        where n_windows_total = n_trials * windows_per_trial.
    """
    config.validate()

    def transform(X: np.ndarray) -> np.ndarray:
        n_trials, n_channels, n_times = X.shape
        all_windows = []

        for trial in X:
            trial_windows = 0
            start = 0
            while start + config.window_samples <= n_times:
                window = trial[:, start:start + config.window_samples]
                all_windows.append(window)
                start += config.step_samples
                trial_windows += 1

            if trial_windows != config.windows_per_trial:
                raise RuntimeError(
                    f"Window count mismatch: expected {config.windows_per_trial}, "
                    f"got {trial_windows}. Actual n_times={n_times}, "
                    f"config.n_times={config.n_times}."
                )

        return np.array(all_windows)

    return transform


def make_stft_fn(
    sfreq: float,
    nperseg: int = 64,
    noverlap: int = 48
) -> Callable[[np.ndarray], np.ndarray]:
    """Create an STFT transform function.

    Operates on already-windowed data. Applies STFT independently per window.

    Args:
        sfreq: Sampling frequency in Hz.
        nperseg: Length of each STFT segment.
        noverlap: STFT segment overlap.

    Returns:
        Transform function that takes (n_windows, n_channels, n_samples) array
        and returns (n_windows, n_channels, n_freqs, n_stft_times) magnitude array.
    """
    def transform(X: np.ndarray) -> np.ndarray:
        all_specs = []
        for window in X:
            _, _, Zxx = stft(window, fs=sfreq, nperseg=nperseg,
                             noverlap=noverlap, axis=-1)
            all_specs.append(np.abs(Zxx))
        return np.array(all_specs)

    return transform


def make_sliding_window_stft_fn(
    config: SlidingWindowConfig,
    stft_nperseg: int = 64,
    stft_overlap: int = 48
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a combined sliding window + STFT transform function.

    Convenience composition of make_sliding_window_fn and make_stft_fn.

    Args:
        config: Sliding window configuration.
        stft_nperseg: Length of each STFT segment.
        stft_overlap: STFT window overlap.

    Returns:
        Transform function that takes (n_trials, n_channels, n_times) array
        and returns (n_windows_total, n_channels, n_freqs, n_stft_times) array.
    """
    window_fn = make_sliding_window_fn(config)
    stft_fn = make_stft_fn(config.sfreq, stft_nperseg, stft_overlap)

    def transform(X: np.ndarray) -> np.ndarray:
        return stft_fn(window_fn(X))

    return transform


def make_label_expander_fn(config: SlidingWindowConfig) -> Callable[[np.ndarray], np.ndarray]:
    """Create a label expansion function for sliding windows.

    Args:
        config: Sliding window configuration (uses windows_per_trial).

    Returns:
        Transform function that repeats each label windows_per_trial times.
    """
    windows_per_trial = config.windows_per_trial

    def transform(events: np.ndarray) -> np.ndarray:
        return np.repeat(events, windows_per_trial, axis=0)

    return transform


def inject_array_transforms(
    process_pipeline: Pipeline,
    x_transform: Callable[[np.ndarray], np.ndarray],
    events_transform: Callable[[np.ndarray], np.ndarray],
    x_name: str = "custom_x",
    events_name: str = "custom_events"
):
    """Inject custom transforms into MOABB's array processing stage.

    Modifies the pipeline in-place by appending transforms to both the X and events
    branches of the ForkPipelines at the array stage.

    Args:
        process_pipeline: MOABB process pipeline from paradigm.make_process_pipelines().
        x_transform: Function to apply to the data array.
        events_transform: Function to apply to the events array (for label sync).
        x_name: Name for the X transform step.
        events_name: Name for the events transform step.

    Raises:
        ValueError: If no array stage with ForkPipelines is found.
    """
    for i, (step_type, transformer) in enumerate(process_pipeline.steps):
        if step_type == StepType.ARRAY and isinstance(transformer, ForkPipelines):
            original_transformers = dict(transformer.transformers)

            # Extend X pipeline
            x_pipeline = original_transformers["X"]
            x_pipeline.steps.append((x_name, FunctionTransformer(x_transform, validate=False)))

            # Extend events pipeline
            events_pipeline = original_transformers["events"]
            new_events_pipeline = Pipeline([
                ("original", events_pipeline),
                (events_name, FunctionTransformer(events_transform, validate=False))
            ])

            # Replace ForkPipelines
            process_pipeline.steps[i] = (
                StepType.ARRAY,
                ForkPipelines(transformers=[("X", x_pipeline), ("events", new_events_pipeline)])
            )
            return

    raise ValueError(
        "Could not find ARRAY stage with ForkPipelines in process_pipeline. "
        "Ensure you're passing a pipeline from paradigm.make_process_pipelines()."
    )


def inject_sliding_window(
    process_pipeline: Pipeline,
    sfreq: float,
    window_sec: float,
    window_overlap: float,
    n_times: int
) -> SlidingWindowConfig:
    """Inject sliding window transform (no STFT) into MOABB pipeline.

    Produces raw windowed signal: (n_windows, n_channels, window_samples).

    Args:
        process_pipeline: MOABB process pipeline (modified in-place).
        sfreq: Sampling frequency in Hz.
        window_sec: Window length in seconds.
        window_overlap: Window overlap ratio [0.0 to 0.99].
        n_times: Number of time samples per trial.

    Returns:
        SlidingWindowConfig used for the transforms.
    """
    config = SlidingWindowConfig(
        sfreq=sfreq,
        window_sec=window_sec,
        window_overlap=window_overlap,
        n_times=n_times,
    )
    config.validate()

    window_fn = make_sliding_window_fn(config)
    expand_fn = make_label_expander_fn(config)

    inject_array_transforms(
        process_pipeline,
        x_transform=window_fn,
        events_transform=expand_fn,
        x_name="sliding_window",
        events_name="expand_labels"
    )

    return config


def inject_sliding_window_stft(
    process_pipeline: Pipeline,
    sfreq: float,
    window_sec: float,
    window_overlap: float,
    n_times: int,
    stft_nperseg: int = 64,
    stft_overlap: int = 48
) -> SlidingWindowConfig:
    """Inject sliding window STFT transform into MOABB pipeline and I am not sorry.

    Convenience function that creates the STFT and label expansion transforms
    and injects them into the pipeline. Both transforms share the same config,
    guaranteeing consistency between data and labels.

    Args:
        process_pipeline: MOABB process pipeline (modified in-place).
        sfreq: Sampling frequency in Hz.
        window_sec: Window length in seconds.
        window_overlap: Window overlap ratio [0.0 to 0.99].
        n_times: Number of time samples per trial (sfreq * epoch_duration) <- varies per dataset.
        stft_nperseg: Length of each STFT segment.
        stft_overlap: STFT window overlap.

    Returns:
        SlidingWindowConfig used for the transforms (access .windows_per_trial etc.).

    Example:
        paradigm = MotorImagery(fmin=4, fmax=40, resample=250, tmin=0, tmax=6)
        process_pipeline = paradigm.make_process_pipelines(dataset)[0]
        config = inject_sliding_window_stft(
            process_pipeline,
            sfreq=250,
            window_sec=3.0,
            window_overlap=0.95,
            n_times=int(6 * 250),  # 6 sec at 250 Hz
        )
        X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=None,
        process_pipelines=[process_pipeline], cache_config=cache_config)

        Example shapes of X, labels, metadata: (52248, 22, 33, 48), (52248,), (52248, 3)
        Where 52248 = 21 windows * 2488 original trials, 22 is eeg channels,
        and 65 x 13 are freq x time STFT spectrograms applied to fix sized windows.
    """
    config = SlidingWindowConfig(
        sfreq=sfreq,
        window_sec=window_sec,
        window_overlap=window_overlap,
        n_times=n_times,
    )
    config.validate()

    stft_fn = make_sliding_window_stft_fn(config, stft_nperseg, stft_overlap)
    expand_fn = make_label_expander_fn(config)

    inject_array_transforms(
        process_pipeline,
        x_transform=stft_fn,
        events_transform=expand_fn,
        x_name="sliding_window_stft",
        events_name="expand_labels"
    )

    return config