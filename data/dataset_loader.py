"""
Data loading and preprocessing for UCR/UEA time series datasets.

Datasets (Table 1):
  Univariate:  CBF, TwoLeadECG, GunPoint, Earthquakes, Coffee, ItalyPowerDemand
  Multivariate: Cricket, Epilepsy

Uses `aeon` for automatic downloading from the UCR/UEA archives.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

# ── Auto-download via aeon ───────────────────────────────────────────────────
try:
    from aeon.datasets import load_classification
    HAS_AEON = True
except ImportError:
    HAS_AEON = False
    print("[WARNING] aeon not installed. Run: pip install aeon")


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapper for time series classification data.
    
    Stores data as tensors of shape (n_samples, n_channels, seq_len)
    and integer class labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: numpy array of shape (n_samples, n_channels, seq_len)
            y: numpy array of integer labels (n_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(name: str):
    """
    Load a dataset by name from the UCR/UEA archive.
    
    Returns:
        X_train: np.ndarray of shape (n_train, d, T)
        y_train: np.ndarray of int labels (n_train,)
        X_test:  np.ndarray of shape (n_test, d, T)
        y_test:  np.ndarray of int labels (n_test,)
        metadata: dict with keys 'n_classes', 'seq_len', 'n_channels'
    """
    if not HAS_AEON:
        raise ImportError("aeon is required for dataset loading. Install via: pip install aeon")

    print(f"[DATA] Loading dataset: {name} ...")

    # Load from aeon — returns (X, y) where X has shape (n, n_channels, seq_len)
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    # aeon returns X as np.ndarray of shape (n_samples, n_channels, seq_len)
    # y is returned as string array — convert to integer
    # First build a label map
    all_labels = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    label_map = {label: idx for idx, label in enumerate(all_labels)}
    y_train = np.array([label_map[l] for l in y_train])
    y_test = np.array([label_map[l] for l in y_test])

    # Handle NaN values (some UCR datasets have trailing NaN for unequal length)
    X_train = np.nan_to_num(X_train, nan=0.0).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0).astype(np.float32)

    # Normalise each channel independently (z-score) using train stats
    for ch in range(X_train.shape[1]):
        mean = X_train[:, ch, :].mean()
        std = X_train[:, ch, :].std()
        if std < 1e-8:
            std = 1.0
        X_train[:, ch, :] = (X_train[:, ch, :] - mean) / std
        X_test[:, ch, :] = (X_test[:, ch, :] - mean) / std

    n_classes = len(all_labels)
    seq_len = X_train.shape[2]
    n_channels = X_train.shape[1]

    metadata = {
        "n_classes": n_classes,
        "seq_len": seq_len,
        "n_channels": n_channels,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_map": label_map,
    }

    print(f"  -> n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"T={seq_len}, d={n_channels}, c={n_classes}")

    return X_train, y_train, X_test, y_test, metadata


def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders from numpy arrays."""
    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader


def get_target_class_samples(X_train, y_train, target_class, device="cpu"):
    """
    Get all training samples belonging to a specific class.
    
    Returns:
        torch.Tensor of shape (n_target, n_channels, seq_len)
    """
    mask = y_train == target_class
    X_target = X_train[mask]
    return torch.tensor(X_target, dtype=torch.float32).to(device)
