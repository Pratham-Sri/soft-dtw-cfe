"""CMA-ES black-box optimizer wrapper (requires `cma`)."""

import numpy as np


def optimize(fn, x0, sigma0=0.5, maxfevals=None, popsize=None,
             seed=None, verbose=False):
    """
    Minimize fn(theta) with CMA-ES.

    Args:
        fn: callable theta (D,) -> float
        x0: (D,) initial mean (use zeros for DTW-centered start)
        sigma0: initial step size
        maxfevals: budget; default 100*D
        popsize: CMA popsize; default 4+int(3*log(D))
        seed: int or None
        verbose: log per-generation best

    Returns:
        dict with xbest, fbest, nfev, best_history
    """
    try:
        import cma
    except ImportError as e:
        raise ImportError("cma required. pip install cma") from e

    x0 = np.asarray(x0, dtype=float).ravel()
    D = x0.size
    if maxfevals is None:
        maxfevals = 100 * D
    opts = {"maxfevals": int(maxfevals), "verbose": -9, "seed": seed}
    if popsize is not None:
        opts["popsize"] = int(popsize)

    nfev = [0]

    def safe_fn(theta):
        nfev[0] += 1
        try:
            v = float(fn(np.asarray(theta, dtype=float)))
        except Exception:
            return 1e12
        if not np.isfinite(v):
            return 1e12
        return v

    def _es_evals(es):
        for attr in ("countevaluations", "count_evaluations",
                     "countevals", "evaluations"):
            v = getattr(es, attr, None)
            if isinstance(v, (int, float, np.integer)):
                return int(v)
        res = getattr(es, "result", None)
        for attr in ("evaluations", "evals", "nfev"):
            v = getattr(res, attr, None) if res is not None else None
            if isinstance(v, (int, float, np.integer)):
                return int(v)
        return nfev[0]

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    best_hist = []
    while not es.stop():
        sols = es.ask()
        vals = [safe_fn(s) for s in sols]
        es.tell(sols, vals)
        best_hist.append(float(np.min(vals)))
        if verbose:
            print(f"  [CMA] fev={_es_evals(es)} best={best_hist[-1]:.4f}")
    res = es.result
    xbest = np.asarray(res.xbest, dtype=float)
    try:
        fbest = float(res.fbest)
    except Exception:
        fbest = safe_fn(xbest)
    return {"xbest": xbest, "fbest": fbest, "nfev": _es_evals(es),
            "best_history": best_hist}
