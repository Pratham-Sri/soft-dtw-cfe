"""DTW path computation (numpy + optional numba/tslearn acceleration)."""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    from tslearn.metrics import dtw_path as _tslearn_dtw_path
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False


def _to_channels_first(x):
    """Accept (d,T) or (T,) or torch tensor, return np (d,T)."""
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def compute_cost_matrix(x, y):
    """
    Squared-Euclidean cost summed over channels.

    Args:
        x: (d, T1), y: (d, T2)
    Returns:
        cost: (T1, T2), cost[i,j] = ||x[:,i]-y[:,j]||^2
    """
    x = _to_channels_first(x)  # (d, T1)
    y = _to_channels_first(y)  # (d, T2)
    # (T1, d) vs (T2, d)
    xt = x.T
    yt = y.T
    diff = xt[:, None, :] - yt[None, :, :]  # (T1, T2, d)
    return np.sum(diff ** 2, axis=-1)


if HAS_NUMBA:
    @njit(fastmath=True)
    def _fill_full(cost, R):
        T1, T2 = cost.shape
        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                a = R[i - 1, j - 1]
                b = R[i - 1, j]
                if b < a:
                    a = b
                c = R[i, j - 1]
                if c < a:
                    a = c
                R[i, j] = cost[i - 1, j - 1] + a

    @njit(fastmath=True)
    def _fill_banded(cost, R, w):
        T1, T2 = cost.shape
        for i in range(1, T1 + 1):
            j_start = i - w
            if j_start < 1:
                j_start = 1
            j_end = i + w
            if j_end > T2:
                j_end = T2
            for j in range(j_start, j_end + 1):
                a = R[i - 1, j - 1]
                b = R[i - 1, j]
                if b < a:
                    a = b
                c = R[i, j - 1]
                if c < a:
                    a = c
                R[i, j] = cost[i - 1, j - 1] + a


def compute_dtw_path(x, y, window=None):
    """
    Classical DTW with backtracked path. Single shared path over channels.

    Args:
        x: (d, T1) — original X
        y: (d, T2) — prototype P
        window: Sakoe-Chiba band radius or None for full matrix.
            For T>500 use window=max(50, T//10) to cut O(T^2)->O(T*w).

    Returns:
        path: (L, 2) int array of (i_X, j_P) from (0,0) to (T1-1,T2-1)
        dist: float accumulated cost along path
    """
    x = _to_channels_first(x)
    y = _to_channels_first(y)
    T1, T2 = x.shape[1], y.shape[1]

    cost = compute_cost_matrix(x, y)  # (T1, T2)

    # Fast path: tslearn C implementation when full matrix, no band.
    if HAS_TSLEARN and window is None and T1 == T2:
        xt = x.T  # (T, d)
        yt = y.T
        path_list, dist = _tslearn_dtw_path(xt, yt)
        return np.array(path_list, dtype=np.int64), float(dist)

    INF = 1e18
    R = np.full((T1 + 1, T2 + 1), INF, dtype=np.float64)
    R[0, 0] = 0.0

    if HAS_NUMBA:
        if window is None:
            _fill_full(cost, R)
        else:
            _fill_banded(cost, R, max(int(window), abs(T1 - T2)))
    elif window is None:
        for i in range(1, T1 + 1):
            ci = cost[i - 1]
            Ri, Rim1 = R[i], R[i - 1]
            for j in range(1, T2 + 1):
                Rim1_j = Rim1[j]
                a = Rim1[j - 1]
                if Rim1_j < a:
                    a = Rim1_j
                b = Ri[j - 1]
                if b < a:
                    a = b
                Ri[j] = ci[j - 1] + a
    else:
        w = max(int(window), abs(T1 - T2))
        for i in range(1, T1 + 1):
            ci = cost[i - 1]
            Ri, Rim1 = R[i], R[i - 1]
            j_start = max(1, i - w)
            j_end = min(T2, i + w)
            for j in range(j_start, j_end + 1):
                Rim1_j = Rim1[j]
                a = Rim1[j - 1]
                if Rim1_j < a:
                    a = Rim1_j
                b = Ri[j - 1]
                if b < a:
                    a = b
                Ri[j] = ci[j - 1] + a

    dist = float(R[T1, T2])

    # Backtrack
    i, j = T1, T2
    rev = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            a = R[i - 1, j - 1]
            b = R[i - 1, j]
            c = R[i, j - 1]
            if a <= b and a <= c:
                i -= 1
                j -= 1
            elif b <= a and b <= c:
                i -= 1
            else:
                j -= 1
        rev.append((i - 1, j - 1))
    path = np.array(rev[::-1], dtype=np.int64)
    return path, dist


def compute_dtw_distance(x, y, window=None):
    """Distance only wrapper."""
    _, dist = compute_dtw_path(x, y, window=window)
    return dist


def compute_dtw_to_prototypes(x, prototypes, window=None):
    """
    Args:
        x: (d, T)
        prototypes: (K, d, T) or list of (d, T)
    Returns:
        list of (path, dist) per prototype
    """
    out = []
    for k in range(len(prototypes)):
        out.append(compute_dtw_path(x, prototypes[k], window=window))
    return out
