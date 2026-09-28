"""DTW-guided constrained deformation CFE (ties all modules together)."""

import numpy as np

from .dtw import compute_dtw_path
from .warp import build_tau_dtw
from .basis import build_amplitude_basis, build_temporal_basis
from .deformation import DeformationGenerator
from .prototypes import select_prototypes
from .cmaes_search import optimize


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


class DTWGuidedCFE:
    """
    Generate CFEs via low-dim DTW-centered deformation + CMA-ES.

    Result dict matches M-CELS/proposed_method keys so
    run_experiments.py and evaluation/metrics.py work unchanged.
    """

    def __init__(self, classifier, M_a=4, M_t=4, K=3, lambda_=1.0,
                 lambda_r=1e-3, m0=0.0, sigma0=0.5, maxfevals=None,
                 popsize=None, seed=0, window=None, width_factor=1.0,
                 eps=1e-3):
        self.classifier = classifier
        self.M_a = M_a
        self.M_t = M_t
        self.K = K
        self.lambda_ = lambda_
        self.lambda_r = lambda_r
        self.m0 = m0
        self.sigma0 = sigma0
        self.maxfevals = maxfevals
        self.popsize = popsize
        self.seed = seed
        self.window = window
        self.width_factor = width_factor
        self.eps = eps
        try:
            import torch
            try:
                self.device = next(classifier.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        except Exception:
            self.device = None

    def _logits(self, x_np):
        import torch
        self.classifier.eval()
        with torch.no_grad():
            xb = torch.tensor(x_np.astype(np.float32)[None, :, :]
                              ).to(self.device)
            lg = self.classifier(xb)
            return _to_numpy(lg)[0]

    def _margin_pred(self, x_np, target):
        lg = self._logits(x_np)
        zt = lg[target]
        m = float(zt - np.max(np.delete(lg, target)))
        return m, int(np.argmax(lg)), lg

    def generate(self, x, target_class, target_samples,
                 X_train=None, y_train=None, verbose=False):
        import torch
        Xn = _to_numpy(x)  # (d,T)
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)
        d, T = Xn.shape
        target = int(target_class)

        with torch.no_grad() if hasattr(torch, "no_grad") else _dummy_ctx():
            try:
                om, orig_pred, _ = self._margin_pred(Xn, target)
            except Exception:
                orig_pred = target

        # Prototype pool: prefer full train arrays when given.
        if X_train is not None and y_train is not None:
            pool_X, pool_y = np.asarray(X_train), np.asarray(y_train)
        else:
            pool_X = _to_numpy(target_samples)
            if pool_X.ndim == 2:
                pool_X = pool_X[None, :, :]
            pool_y = np.full(len(pool_X), target)

        protos = select_prototypes(Xn, target, pool_X, pool_y,
                                   self.classifier, K=self.K,
                                   device=self.device)
        if not protos:
            return {"counterfactual": torch.tensor(Xn.astype(np.float32)),
                    "original": torch.tensor(Xn.astype(np.float32)),
                    "target_class": target, "original_class": orig_pred,
                    "predicted_class": orig_pred, "valid": False,
                    "losses": {"total": []}, "margin": float("-inf"),
                    "distance": 0.0}

        window = self.window
        if window is None and T > 500:
            window = max(50, T // 10)

        R_a = build_amplitude_basis(T, self.M_a,
                                    width_factor=self.width_factor)
        R_t = build_temporal_basis(T, self.M_t,
                                   width_factor=self.width_factor)
        D = self.M_a + self.M_t

        best, best_key = None, None
        for pi, pr in enumerate(protos):
            P = np.asarray(pr["prototype"], dtype=np.float64)
            path, _ = compute_dtw_path(Xn, P, window=window)
            w = build_tau_dtw(path, T, T_out=T, eps=self.eps)
            gen = DeformationGenerator(R_a, R_t, w["g_dtw"]).set_pair(Xn, P)

            def fn(theta, _gen=gen):
                Xp = _gen(theta)
                Dm = float(np.sum((Xp - Xn) ** 2) / Xn.size)
                m, _, _ = self._margin_pred(Xp, target)
                return Dm + self.lambda_ * max(0.0, self.m0 - m) \
                    + self.lambda_r * float(np.sum(np.asarray(theta) ** 2))

            res = optimize(fn, np.zeros(D), sigma0=self.sigma0,
                           maxfevals=self.maxfevals, popsize=self.popsize,
                           seed=(self.seed or 0) + pi, verbose=verbose)
            Xb = gen(res["xbest"])
            Db = float(np.sum((Xb - Xn) ** 2) / Xn.size)
            mb, predb, _ = self._margin_pred(Xb, target)
            valid = predb == target
            key = (1 if valid else 0, -Db if valid else mb)
            if best_key is None or key > best_key:
                best_key = key
                best = {"Xb": Xb, "Db": Db, "mb": mb, "pred": predb,
                        "valid": valid, "hist": res["best_history"],
                        "nfev": res["nfev"], "pidx": pr["index"]}

        return {"counterfactual": torch.tensor(best["Xb"].astype(np.float32)),
                "original": torch.tensor(Xn.astype(np.float32)),
                "target_class": target, "original_class": orig_pred,
                "predicted_class": best["pred"], "valid": best["valid"],
                "losses": {"total": best["hist"]},
                "margin": best["mb"], "distance": best["Db"],
                "prototype_index": best["pidx"], "nfev": best["nfev"]}

    def generate_batch(self, X, y, X_train, y_train, verbose=True):
        import torch
        results = []
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        for i in range(n):
            xi = X[i]
            yi = y[i].item() if hasattr(y[i], "item") else int(y[i])
            with torch.no_grad():
                try:
                    probs = self._logits(_to_numpy(xi))
                    pc = probs.copy()
                    pc[yi] = -1e18
                    tc = int(np.argmax(pc))
                except Exception:
                    tc = (yi + 1) % len(np.unique(np.asarray(y_train)))
            # target pool for compat with other methods' generate()
            mask = (np.asarray(y_train) == tc)
            import torch as _t
            tpool = _t.tensor(np.asarray(X_train)[mask], dtype=_t.float32)
            r = self.generate(xi, tc, tpool, X_train, y_train, verbose=False)
            results.append(r)
            if verbose and (i + 1) % 10 == 0:
                print(f"  [DTW-CFE] [{i+1}/{n}] class {yi} -> {tc} "
                      f"valid={r['valid']}")
        return results


class _dummy_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
