import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def collate_spectrograms(batch):
    """Custom collate that pads ch x H x W spectrograms and returns lengths."""
    signals, labels = zip(*batch)
    lengths = torch.tensor([s.shape[2] for s in signals])

    signals_permuted = [s.permute(2, 0, 1) for s in signals]  # W x ch x H
    padded = pad_sequence(signals_permuted, batch_first=True, padding_value=0) # B x W x ch x H
    padded = padded.permute(0, 2, 3, 1) # B x ch x H x W

    labels = torch.tensor(labels)
    return padded, labels, lengths

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, stratify=True, seed=4269):
    """Split dataset into train/val/test sets with torch methods.

    Args:
        dataset: Torch dataset.
        train_ratio: Test split ratio.
        val_ratio: Validation split ratio.
        stratify: Whether to preserve class distribution.
        seed: Randomness seed.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    if stratify:
        indices = list(range(len(dataset)))
        labels = [dataset[i][1] for i in indices]

        # train / val+test split
        train_idx, tmp_idx = train_test_split(indices, train_size=train_ratio, stratify=labels, random_state=seed)

        # If just train / test
        if val_ratio == 0.0:
            return Subset(dataset, train_idx), None, Subset(dataset, tmp_idx)
        # If just train / val
        if train_ratio + val_ratio == 1.0:
            return Subset(dataset, train_idx), Subset(dataset, tmp_idx), None

        # val / test split
        tmp_labels = [labels[i] for i in tmp_idx]
        val_idx, test_idx = train_test_split(
            tmp_idx, train_size=val_ratio / (1 - train_ratio), stratify=tmp_labels, random_state=seed
        )

        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    generator = torch.Generator().manual_seed(seed)

    n = len(dataset)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size], generator)

def kfold_splits(dataset, n_folds=5, stratify=True, seed=4269):
    """Generator that yields (train_subset, val_subset) for each fold.

    Args:
        dataset: Torch dataset.
        n_folds: Numbrer of folds
        stratify: Whether to preserve class distribution.
        seed: Randomness seed.
    """
    indices = list(range(len(dataset)))
    labels = [dataset[i][1] for i in indices]

    splitter = (StratifiedKFold if stratify else KFold)(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = splitter.split(indices, labels)

    for train_idx, val_idx in splits:
        yield Subset(dataset, train_idx), Subset(dataset, val_idx)
