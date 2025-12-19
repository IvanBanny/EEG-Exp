"""Torch Dataset wrapper for MOABB MI data.

This module provides a Torch-compatible Dataset class that wraps any MOABB
dataset and paradigm pipeline, enabling integration with Torch DataLoaders.

Example:
    from torch.utils.data import DataLoader
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    from src.train_utils import STFTDataset, STFTDatasetConfig

    dataset = BNCI2014_001()
    paradigm = MotorImagery(fmin=4, fmax=40, resample=250, tmin=0, tmax=4)

    config = STFTDatasetConfig(
        paradigm=paradigm,
        window_sec=2.0,
        window_overlap=0.9,
    )
    torch_dataset = STFTDataset(dataset, config)
    loader = DataLoader(torch_dataset, batch_size=64, shuffle=True, num_workers=4)

    for X, y in loader:
        # X: (batch, channels, freqs, times)
        # y: (batch,)
        ...
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm

from ..data_proc import inject_sliding_window_stft


@dataclass
class STFTDatasetConfig:
    """Configuration for STFTDataset.

    Centralizes preprocessing and windowing parameters. The paradigm object
    defines filtering, resampling, and epoch extraction. This config handles
    the sliding window STFT stage and caching.

    Attributes:
        paradigm: MOABB paradigm instance (e.g., MotorImagery). Defines fmin,
            fmax, resample, tmin, tmax, events, etc.
        subjects: List of subject IDs to load. None means all subjects.
        window_sec: Sliding window length in seconds.
        window_overlap: Window overlap ratio in [0.0, 0.99].
        stft_nperseg: STFT segment length.
        stft_overlap: STFT segment overlap.
        cache_path: Path for MOABB cache. None disables caching.
        use_cache: Whether to use MOABB's array caching.
        dtype: numpy dtype for data arrays.
    """

    paradigm: BaseParadigm
    subjects: Optional[list[int]] = None
    window_sec: float = 2.0
    window_overlap: float = 0.9
    stft_nperseg: int = 64
    stft_overlap: int = 48
    cache_path: Optional[str] = "./moabb_cache"
    use_cache: bool = True
    dtype: np.dtype = np.float32

    def get_n_times(self, dataset: BaseDataset) -> int:
        """Compute number of time samples per epoch for a given dataset.

        Uses the paradigm's tmin/tmax and resample settings along with
        dataset-specific interval information.

        Args:
            dataset: MOABB dataset to compute n_times for.

        Returns:
            Number of time samples per epoch after resampling.
        """
        # Paradigm settings take precedence if set
        tmin = self.paradigm.tmin if self.paradigm.tmin is not None else dataset.interval[0]
        tmax = self.paradigm.tmax if self.paradigm.tmax is not None else dataset.interval[1]
        sfreq = self.paradigm.resample if self.paradigm.resample is not None else 250.0

        duration = tmax - tmin
        return int(duration * sfreq)

    def get_sfreq(self) -> float:
        """Get the sampling frequency after paradigm resampling.

        Returns:
            Sampling frequency in Hz.

        Raises:
            ValueError: If paradigm.resample is not set.
        """
        if self.paradigm.resample is None:
            raise ValueError(
                "paradigm.resample must be set for sliding window STFT. "
                "The transform requires a known, fixed sampling rate."
            )
        return float(self.paradigm.resample)


class STFTDataset(Dataset):
    """Torch Dataset for Motor Imagery EEG data.

    Wraps any MOABB dataset and paradigm pipeline to provide windowed STFT
    spectrograms as torch tensors. Data is loaded once at initialization
    and stored in memory for fast access during training.

    The class handles:
        - Pipeline injection for sliding window STFT
        - Label encoding (string -> integer)
        - Conversion to torch tensors

    Attributes:
        config: Dataset configuration.
        data: Preprocessed EEG data as tensor (n_windows, n_channels, n_freqs, n_times).
        labels: Integer class labels as tensor (n_windows,).
        metadata: Original MOABB metadata DataFrame.
        label_map: Mapping from class names to integer indices.
        window_config: Sliding window configuration used for preprocessing.
    """

    def __init__(self, dataset: BaseDataset, config: STFTDatasetConfig) -> None:
        """Initialize the dataset.

        Loads and preprocesses all data according to the configuration.
        This may take a while on first run without cache.

        Args:
            dataset: Any MOABB-compatible dataset instance.
            config: Dataset configuration object.
        """
        self.config = config
        self.moabb_dataset = dataset
        self._load_data()

    def _load_data(self) -> None:
        """Load and preprocess data using MOABB pipeline."""
        paradigm = self.config.paradigm
        dataset = self.moabb_dataset

        process_pipeline = paradigm.make_process_pipelines(dataset)[0]

        sfreq = self.config.get_sfreq()
        n_times = self.config.get_n_times(dataset)

        self.window_config = inject_sliding_window_stft(
            process_pipeline,
            sfreq=sfreq,
            window_sec=self.config.window_sec,
            window_overlap=self.config.window_overlap,
            n_times=n_times,
            stft_nperseg=self.config.stft_nperseg,
            stft_overlap=self.config.stft_overlap,
        )

        cache_config = self._make_cache_config()

        X, labels, metadata = paradigm.get_data(
            dataset=dataset,
            subjects=self.config.subjects,
            process_pipelines=[process_pipeline],
            cache_config=cache_config,
        )

        # Encode string labels to integers
        unique_labels = sorted(set(labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = np.array([self.label_map[lbl] for lbl in labels], dtype=np.int64)

        # Convert to tensors
        self.data = torch.from_numpy(X.astype(self.config.dtype))
        self.labels = torch.from_numpy(encoded_labels)
        self.metadata = metadata

    def _make_cache_config(self) -> dict:
        """Create MOABB cache configuration dictionary."""
        if not self.config.use_cache or self.config.cache_path is None:
            return {"use": False}

        return {
            "use": True,
            "path": self.config.cache_path,
            "save_raw": False,
            "save_epochs": False,
            "save_array": True,
            "overwrite_raw": False,
            "overwrite_epochs": False,
            "overwrite_array": False,
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (data, label) tensors.
            data: (n_channels, n_freqs, n_times) spectrogram.
            label: Scalar integer class label.
        """
        return self.data[idx], self.labels[idx]

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return self.data.shape[1]

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins in STFT output."""
        return self.data.shape[2]

    @property
    def n_stft_times(self) -> int:
        """Number of time steps in STFT output."""
        return self.data.shape[3]

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Shape of a single input sample (channels, freqs, times)."""
        return self.n_channels, self.n_freqs, self.n_stft_times

    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(self.label_map)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data.

        Returns:
            Tensor of shape (n_classes,) with weight for each class.
        """
        _, counts = torch.unique(self.labels, return_counts=True)
        weights = 1.0 / counts.float()
        return weights / weights.sum() * len(weights)

    def get_subject_indices(self, subject_id: int) -> np.ndarray:
        """Get sample indices belonging to a specific subject.

        Useful for subject-wise cross-validation splits.

        Args:
            subject_id: Subject ID to filter by.

        Returns:
            Array of indices for samples from the specified subject.
        """
        mask = self.metadata["subject"].values == subject_id
        return np.where(mask)[0]

    def get_subject_ids(self) -> list[int]:
        """Get list of unique subject IDs in the dataset.

        Returns:
            Sorted list of subject IDs present in the loaded data.
        """
        return sorted(self.metadata["subject"].unique().tolist())

    def split_by_subject(
        self, train_subjects: list[int], test_subjects: list[int]
    ) -> tuple["SubsetSTFTDataset", "SubsetSTFTDataset"]:
        """Create train/test splits based on subject IDs.

        This is the proper way to evaluate generalization across subjects,
        avoiding data leakage from windowed samples of the same trial.

        Args:
            train_subjects: Subject IDs for training set.
            test_subjects: Subject IDs for test set.

        Returns:
            Tuple of (train_dataset, test_dataset) as SubsetSTFTDataset objects.
        """
        train_mask = self.metadata["subject"].isin(train_subjects).values
        test_mask = self.metadata["subject"].isin(test_subjects).values

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        return (
            SubsetSTFTDataset(self, train_indices),
            SubsetSTFTDataset(self, test_indices),
        )

    def split_by_session(
        self, train_sessions: list[str], test_sessions: list[str]
    ) -> tuple["SubsetSTFTDataset", "SubsetSTFTDataset"]:
        """Create train/test splits based on session IDs.

        Useful for within-subject cross-session evaluation.

        Args:
            train_sessions: Session IDs for training set.
            test_sessions: Session IDs for test set.

        Returns:
            Tuple of (train_dataset, test_dataset) as SubsetSTFTDataset objects.
        """
        train_mask = self.metadata["session"].isin(train_sessions).values
        test_mask = self.metadata["session"].isin(test_sessions).values

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        return (
            SubsetSTFTDataset(self, train_indices),
            SubsetSTFTDataset(self, test_indices),
        )


class SubsetSTFTDataset(Dataset):
    """A subset view of STFTDataset.

    Provides a Dataset interface over a subset of indices from the parent
    STFTDataset. Does not copy data, just indexes into the parent.

    Attributes:
        parent: The parent STFTDataset.
        indices: Array of valid indices into the parent dataset.
    """

    def __init__(self, parent: STFTDataset, indices: np.ndarray) -> None:
        """Initialize subset dataset.

        Args:
            parent: Parent STFTDataset to take subset from.
            indices: Array of indices to include in this subset.
        """
        self.parent = parent
        self.indices = indices

    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by subset index.

        Args:
            idx: Index within this subset (not parent index).

        Returns:
            Tuple of (data, label) tensors from the parent dataset.
        """
        parent_idx = self.indices[idx]
        return self.parent[parent_idx]

    @property
    def data(self) -> torch.Tensor:
        """Subset view of the data tensor."""
        return self.parent.data[self.indices]

    @property
    def labels(self) -> torch.Tensor:
        """Subset view of the labels tensor."""
        return self.parent.labels[self.indices]

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Shape of a single input sample (channels, freqs, times)."""
        return self.parent.input_shape

    @property
    def n_classes(self) -> int:
        """Number of unique classes in the parent dataset."""
        return self.parent.n_classes

    @property
    def label_map(self) -> dict[str, int]:
        """Label mapping from the parent dataset."""
        return self.parent.label_map

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for this subset.

        Returns:
            Tensor of shape (n_classes,) with weight for each class.
        """
        _, counts = torch.unique(self.labels, return_counts=True)
        weights = 1.0 / counts.float()
        return weights / weights.sum() * len(weights)
