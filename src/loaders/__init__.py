from .torch_dataset import EEGDatasetConfig, EEGDataset, SubsetEEGDataset, STFTDatasetConfig, STFTDataset
from .utils import collate_eeg, collate_spectrograms, split_dataset, kfold_splits
from .transforms import TransformWrapper, Compose, ToTensor, ZScoreNormalize, LogCompress, ClipOutliers, MinMaxNormalize
from .augmentation import GaussianNoise, RandomScale, TimeShift, ChannelDropout
from .factory import build_datasets

__all__ = [
    "EEGDatasetConfig", "EEGDataset",
    "SubsetEEGDataset",
    "STFTDatasetConfig", "STFTDataset",  # backward compat aliases
    "collate_eeg", "collate_spectrograms", "split_dataset", "kfold_splits",
    "TransformWrapper", "Compose", "ToTensor", "ZScoreNormalize", "LogCompress", "ClipOutliers", "MinMaxNormalize",
    "GaussianNoise", "RandomScale", "TimeShift", "ChannelDropout",
    "build_datasets",
]