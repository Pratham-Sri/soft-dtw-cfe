"""
Soft-DTW: Differentiable Dynamic Time Warping.

Implements the soft-DTW loss from:
  Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series"
  ICML 2017 (Reference [3] in the paper).

Equation (2) from Section 4:
  DTW_γ(X,Y) = min_γ { ⟨A, Δ(X,Y)⟩ : A ∈ A_{m,m'} }

where min_γ is the soft-minimum:
  min_γ{a₁,...,aₙ} = -γ log Σᵢ exp(-aᵢ/γ)
"""

import torch
import torch.nn as nn
import numpy as np
from tslearn.metrics import SoftDTWLossPyTorch


def _soft_min(a: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Soft-minimum operator (Equation from Section 4).
    
    min_γ{a₁,...,aₙ} = -γ log Σᵢ exp(-aᵢ/γ)
    
    As γ → 0⁺, this converges to the standard minimum.
    """
    return -gamma * torch.logsumexp(-a / gamma, dim=-1)


def pairwise_squared_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances between time steps.
    
    Args:
        x: (T1, d) — first time series
        y: (T2, d) — second time series
    
    Returns:
        cost_matrix: (T1, T2) where entry (i,j) = ||x_i - y_j||²
    """
    # x: (T1, d), y: (T2, d)
    # Using broadcasting: (T1, 1, d) - (1, T2, d) → (T1, T2, d)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return (diff ** 2).sum(dim=-1)


class SoftDTW(nn.Module):
    """
    Differentiable Soft-DTW distance using tslearn's fast implementation.
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.loss = SoftDTWLossPyTorch(gamma=gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Soft-DTW distance between two time series.
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # tslearn expects (batch_size, seq_len, features)
        # Since we compare two series, we add a batch dimension of 1
        d = self.loss(x.unsqueeze(0), y.unsqueeze(0))
        return d.squeeze()


class SoftDTWBatch(nn.Module):
    """
    Batched Soft-DTW computation.
    
    Efficiently computes soft-DTW for multiple pairs using vectorised
    operations along the anti-diagonals of the DP table.
    
    Args:
        gamma: Smoothing parameter (default 1.0).
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self._sdtw = SoftDTW(gamma)

    def forward(self, x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute soft-DTW between x and each series in Y.
        
        Args:
            x: (d, T) — single query (channels-first)
            Y: (K, d, T) — K reference series (channels-first)
        
        Returns:
            distances: (K,) — soft-DTW distances
        """
        # Convert to (T, d) format for the core soft-DTW
        x_td = x.permute(1, 0) if x.dim() == 2 else x.unsqueeze(-1)  # (T, d)
        
        distances = []
        for k in range(Y.shape[0]):
            y_k = Y[k]
            y_td = y_k.permute(1, 0) if y_k.dim() == 2 else y_k.unsqueeze(-1)  # (T, d)
            d = self._sdtw(x_td, y_td)
            distances.append(d)
        
        return torch.stack(distances)


def compute_dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute standard (hard) DTW distance between two time series (numpy).
    
    Used for evaluation metrics (plausibility = avg DTW to k-NN).
    
    Args:
        x: (T, d) or (T,)
        y: (T, d) or (T,)
    
    Returns:
        DTW distance (float)
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    T1, T2 = x.shape[0], y.shape[0]

    # Cost matrix
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # (T1, T2, d)
    cost = np.sum(diff ** 2, axis=-1)  # (T1, T2)

    # DP
    INF = 1e18
    R = np.full((T1 + 1, T2 + 1), INF)
    R[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            R[i, j] = cost[i - 1, j - 1] + min(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1])

    return R[T1, T2]
