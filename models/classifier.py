"""
1D Convolutional Neural Network Classifier for Time Series.

Architecture from Section 5.1:
  Three convolutional blocks (channels: 32, 64, 128), each consisting of:
    - 1D Convolution (kernel_size=3)
    - Batch Normalisation
    - ReLU activation
    - Max Pooling (kernel_size=2)
  Followed by:
    - Adaptive Average Pooling → 1
    - Dropout
    - Linear classification layer

Hyperparameters (dropout, lr, weight_decay) are optimised via Optuna.
Training proceeds for up to 80 epochs with early stopping (patience=10).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

from soft_dtw_cfe.config import DEVICE, CLASSIFIER_CONFIG, MODELS_DIR


class ConvBlock(nn.Module):
    """Single convolutional block: Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class TSClassifier(nn.Module):
    """
    1D CNN for time series classification.
    
    Architecture matches Section 5.1 of the paper exactly:
      3 conv blocks (32 → 64 → 128 channels),
      adaptive average pooling,
      dropout,
      linear classifier.
    
    Args:
        n_channels: Number of input channels (d in the paper).
        n_classes: Number of output classes (c in the paper).
        dropout: Dropout rate (optimised by Optuna).
    """

    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        channels = CLASSIFIER_CONFIG["conv_channels"]  # [32, 64, 128]
        ks = CLASSIFIER_CONFIG["kernel_size"]           # 3

        self.conv_blocks = nn.Sequential(
            ConvBlock(n_channels, channels[0], ks),
            ConvBlock(channels[0], channels[1], ks),
            ConvBlock(channels[1], channels[2], ks),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channels[2], n_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, d, T) — time series input (channels-first).
        Returns:
            logits: (batch, c)
        """
        h = self.conv_blocks(x)
        h = self.pool(h).squeeze(-1)   # (batch, 128)
        h = self.dropout(h)
        return self.fc(h)

    def predict_proba(self, x):
        """Return class probabilities via softmax."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x):
        """Return predicted class labels."""
        return self.forward(x).argmax(dim=-1)


def train_classifier(
    model: TSClassifier,
    train_loader,
    test_loader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = None,
    patience: int = None,
    verbose: bool = True,
):
    """
    Train the classifier with early stopping.
    
    Args:
        model: TSClassifier instance.
        train_loader: Training DataLoader.
        test_loader: Validation/test DataLoader.
        lr: Learning rate (optimised by Optuna).
        weight_decay: L2 regularisation (optimised by Optuna).
        max_epochs: Maximum training epochs (default from config: 80).
        patience: Early stopping patience (default from config: 10).
        verbose: Print progress.
    
    Returns:
        best_model: Model with best validation accuracy.
        best_acc: Best validation accuracy achieved.
        history: dict with 'train_loss', 'val_acc' lists.
    """
    if max_epochs is None:
        max_epochs = CLASSIFIER_CONFIG["max_epochs"]
    if patience is None:
        patience = CLASSIFIER_CONFIG["early_stopping_patience"]

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    best_model_state = None
    no_improve_count = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(max_epochs):
        # ── Training ──
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # ── Validation ──
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch).argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / max(total, 1)
        history["val_acc"].append(val_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{max_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ── Early stopping ──
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_acc, history


def evaluate_classifier(model: TSClassifier, test_loader) -> float:
    """Evaluate classifier accuracy on test set."""
    model.eval()
    model = model.to(DEVICE)
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch).argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / max(total, 1)


def save_classifier(model: TSClassifier, dataset_name: str):
    """Save model weights to disk."""
    path = os.path.join(MODELS_DIR, f"classifier_{dataset_name}.pt")
    torch.save(model.state_dict(), path)
    print(f"  [SAVE] Classifier saved to {path}")
    return path


def load_classifier(model: TSClassifier, dataset_name: str) -> TSClassifier:
    """Load model weights from disk."""
    path = os.path.join(MODELS_DIR, f"classifier_{dataset_name}.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        print(f"  [LOAD] Classifier loaded from {path}")
    else:
        raise FileNotFoundError(f"No saved classifier found at {path}")
    return model
