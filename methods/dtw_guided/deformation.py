"""theta -> tau_theta -> X_warp -> X' (numpy only)."""

import numpy as np

from .warp import softplus


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _to_2d(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def resample_to_length(y, T_out):
    """Linear resample (d,T_in) -> (d,T_out). Identity if equal."""
    y = _to_2d(y)
    d, T_in = y.shape
    if T_in == T_out:
        return y.copy()
    src = np.linspace(0.0, 1.0, T_in)
    dst = np.linspace(0.0, 1.0, T_out)
    out = np.empty((d, T_out))
    for c in range(d):
        out[c] = np.interp(dst, src, y[c])
    return out


class DeformationGenerator:
    """
    Fixed per-(X,P) generator. Precompute once, call per theta.

    tau built from velocity: v=softplus(g_dtw + R_t@beta),
      tau_idx = (T_x-1)*cumsum(v)/sum(v).
    X_warp[c,t] = X[c](tau[t]) via linear interp.
    P_aligned = P resampled to T_out (==P when lengths match; output
      timebase is prototype timebase, X is warped onto it).
    a(t) = sigmoid(R_a@alpha), shared across channels.
    X' = (1-a)*X_warp + a*P_aligned.
    """

    def __init__(self, R_a, R_t, g_dtw):
        self.R_a = np.asarray(R_a, dtype=np.float64)  # (T_out, M_a)
        self.R_t = np.asarray(R_t, dtype=np.float64)  # (T_out, M_t)
        self.g_dtw = np.asarray(g_dtw, dtype=np.float64)  # (T_out,)
        self.T_out = self.R_a.shape[0]

    def set_pair(self, X, P, immut_mask=None):
        """
        X: (d,T_x), P: (d,T_p). immut_mask: (T_out,) bool, True=locked (a=0).
        """
        self.X = _to_2d(X)
        self.T_x = self.X.shape[1]
        self.P_aligned = resample_to_length(P, self.T_out)
        if immut_mask is None:
            self.immut_mask = None
        else:
            self.immut_mask = np.asarray(immut_mask, dtype=bool)
        self._src_idx = np.arange(self.T_x, dtype=np.float64)
        return self

    def theta_to_outputs(self, theta):
        M_a = self.R_a.shape[1]
        theta = np.asarray(theta, dtype=np.float64).ravel()
        alpha = theta[:M_a]
        beta = theta[M_a:]

        g = self.g_dtw + self.R_t @ beta
        v = softplus(g)
        v = np.clip(v, 1e-6, None)
        tau = (self.T_x - 1.0) * np.cumsum(v) / np.sum(v)  # (T_out,)

        d = self.X.shape[0]
        X_warp = np.empty((d, self.T_out))
        for c in range(d):
            X_warp[c] = np.interp(tau, self._src_idx, self.X[c])

        h = self.R_a @ alpha
        a = sigmoid(h)  # (T_out,)
        if self.immut_mask is not None:
            a = np.where(self.immut_mask, 0.0, a)

        X_prime = (1.0 - a)[None, :] * X_warp + a[None, :] * self.P_aligned
        return {"X_prime": X_prime, "X_warp": X_warp, "tau": tau,
                "a": a, "v": v}

    def __call__(self, theta):
        return self.theta_to_outputs(theta)["X_prime"]
