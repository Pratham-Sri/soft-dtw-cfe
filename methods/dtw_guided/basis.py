"""RBF basis for amplitude / temporal deformation (numpy only)."""

import numpy as np


def rbf_matrix(T, M, sigma=None, centers=None, width_factor=1.0):
    """
    Gaussian RBFs over normalized time [0,1].

    Args:
        T: number of time steps
        M: number of basis functions
        sigma: kernel width in normalized units. Default spacing*width_factor.
        centers: (M,) or None for uniform linspace(0,1).
        width_factor: overlap multiplier applied to default spacing.

    Returns:
        R: (T, M), R[t,m] = exp(-(t-c_m)^2 / (2*sigma^2))
    """
    if M < 1:
        raise ValueError("M must be >= 1")
    t = np.linspace(0.0, 1.0, T)
    if centers is None:
        centers = np.linspace(0.0, 1.0, M) if M > 1 else np.array([0.5])
    centers = np.asarray(centers, dtype=np.float64)
    if sigma is None:
        spacing = 1.0 / (M - 1) if M > 1 else 1.0
        sigma = spacing * width_factor
    sigma = max(float(sigma), 1e-6)
    diff = t[:, None] - centers[None, :]
    return np.exp(-diff ** 2 / (2.0 * sigma ** 2))


def build_amplitude_basis(T, M_a, **kwargs):
    """(T, M_a) amplitude basis."""
    return rbf_matrix(T, M_a, **kwargs)


def build_temporal_basis(T, M_t, **kwargs):
    """(T, M_t) temporal basis."""
    return rbf_matrix(T, M_t, **kwargs)
