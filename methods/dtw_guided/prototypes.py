"""Target prototype selection (train-only, correctly-classified, deterministic)."""

import numpy as np


def _batch_logits(model, X_np, device=None, batch=256):
    """X_np: (N,d,T) -> logits (N,C). No-grad, eval mode."""
    import torch
    was_training = model.training
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    out = []
    with torch.no_grad():
        for s in range(0, len(X_np), batch):
            xb = torch.tensor(X_np[s:s + batch], dtype=torch.float32).to(device)
            logits = model(xb)
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()
            else:
                logits = np.asarray(logits)
            out.append(logits)
    if was_training:
        model.train()
    return np.concatenate(out, axis=0) if out else np.empty((0, 0))


def target_margin(logits, target):
    """m = z_t - max_{c!=t} z_c. logits: (...,C)."""
    logits = np.asarray(logits, dtype=np.float64)
    zt = logits[..., target]
    other = np.delete(logits, target, axis=-1)
    return zt - np.max(other, axis=-1)


def select_prototypes(X, target_class, X_train, y_train, model, K=5,
                      device=None):
    """
    Deterministic top-K by L2 distance among correctly classified train samples.

    Args:
        X: (d,T) query (numpy or torch)
        target_class: int
        X_train: (N,d,T) numpy, y_train: (N,) numpy
        model: classifier with forward->logits
        K: number of prototypes

    Returns:
        list of dicts: prototype (d,T), index, margin, distance.
        Sorted by distance ascending. Empty entries excluded.
    """
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    mask = (y_train == int(target_class))
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    cand = X_train[idx]  # (Nc,d,T)

    logits = _batch_logits(model, cand.astype(np.float32), device=device)
    margins = target_margin(logits, int(target_class))
    valid = margins > 0
    if not np.any(valid):
        return []

    vidx = idx[valid]
    vcand = cand[valid]
    vmargins = margins[valid]

    diff = vcand - X[None, :, :]
    dists = np.sum(diff ** 2, axis=(1, 2))

    order = np.argsort(dists, kind="stable")
    out = []
    for r in order[:K]:
        out.append({"prototype": vcand[r].astype(np.float32),
                    "index": int(vidx[r]),
                    "margin": float(vmargins[r]),
                    "distance": float(dists[r])})
    return out
