"""DTW path -> continuous monotonic warp (PCHIP + velocity representation)."""

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def softplus(x):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    large = x > 30.0
    out[large] = x[large]
    xl = x[~large]
    out[~large] = np.log1p(np.exp(xl))
    return out


def inv_softplus(v, v_min=1e-6, v_max=50.0):
    """Inverse softplus with clipping. v > 0."""
    v = np.clip(np.asarray(v, dtype=np.float64), v_min, v_max)
    out = np.empty_like(v)
    large = v > 20.0
    out[large] = v[large]  # softplus(x) ~= x
    vs = v[~large]
    out[~large] = np.log(np.expm1(vs))
    return out


G_IDENTITY_SCALAR = float(inv_softplus(np.array(1.0)))


def path_to_mapping(path, T_x, T_p):
    """
    Convert (i_X, j_P) path to P->X mapping via median aggregation.

    Args:
        path: (L,2) int array (i_X, j_P)
        T_x: length of X, T_p: length of P
    Returns:
        j_grid: (T_p,) arange, i_med: (T_p,) median i_X per j_P
    """
    path = np.asarray(path, dtype=np.int64)
    i_med = np.empty(T_p, dtype=np.float64)
    for j in range(T_p):
        sel = path[path[:, 1] == j, 0]
        if len(sel) == 0:
            # No match: nearest-path fill (should be rare for valid DTW)
            # fall back to linear interpolation between neighbours later
            i_med[j] = np.nan
        else:
            i_med[j] = float(np.median(sel))
    # Fill NaNs by linear interp (monotonic path guarantees few gaps)
    nan = np.isnan(i_med)
    if np.any(nan):
        good = ~nan
        i_med[nan] = np.interp(
            np.flatnonzero(nan), np.flatnonzero(good), i_med[good]
        )
    # Enforce non-decreasing (median of monotonic path is ~monotonic,
    # numerical ties aside)
    i_med = np.maximum.accumulate(i_med)
    i_med = np.clip(i_med, 0, T_x - 1)
    return np.arange(T_p), i_med


def build_tau_dtw(path, T_x, T_out=None, eps=1e-3):
    """
    Build smooth tau_DTW + velocity-centering terms.

    Args:
        path: (L,2) (i_X, j_P)
        T_x: length of X (index space sampled by tau)
        T_out: output length (defaults T_x). When T_x==T_p this is T.
        eps: positive derivative floor in normalized units.

    Returns:
        dict with tau (T_out,) in X-index units [0,T_x-1],
                  v (T_out,) mean-1 positive velocity,
                  g_dtw (T_out,), g_identity (T_out,)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for PCHIP warp. pip install scipy")
    T_p = int(np.max(np.asarray(path)[:, 1]) + 1)
    if T_out is None:
        T_out = T_x

    j_grid, i_med = path_to_mapping(path, T_x, T_p)

    # Normalize to [0,1]
    j_norm = j_grid / max(T_p - 1, 1)
    i_norm = i_med / max(T_x - 1, 1)

    pchip = PchipInterpolator(j_norm, i_norm, extrapolate=True)
    t_norm = np.linspace(0.0, 1.0, T_out)
    tau_norm = np.clip(pchip(t_norm), 0.0, 1.0)
    # Enforce non-decreasing after interpolation noise
    tau_norm = np.maximum.accumulate(tau_norm)
    tau_norm[0], tau_norm[-1] = 0.0, 1.0

    # Derivative -> velocity with floor, rescaled to mean 1
    deriv = np.clip(pchip.derivative()(t_norm), eps, None)
    # Correct for T_x != T_p scale: d(i_norm)/d(j) has mean T_p/T_x-ish;
    # normalize so mean velocity == 1 (recovered by cumsum renormalization).
    deriv = deriv / max(np.mean(deriv), 1e-12)
    deriv = np.clip(deriv, eps, None)
    v = deriv
    g_dtw = inv_softplus(v)
    g_identity = np.full(T_out, G_IDENTITY_SCALAR)

    tau = tau_norm * (T_x - 1)
    return {"tau": tau, "tau_norm": tau_norm, "v": v,
            "g_dtw": g_dtw, "g_identity": g_identity}


def validate_tau(tau, T_x):
    """Return (endpoint_ok, monotonic_ok) for a tau in index units."""
    tau = np.asarray(tau)
    endpoint_ok = abs(tau[0]) < 1e-6 and abs(tau[-1] - (T_x - 1)) < 1e-6
    monotonic_ok = bool(np.all(np.diff(tau) > 0))
    return endpoint_ok, monotonic_ok
