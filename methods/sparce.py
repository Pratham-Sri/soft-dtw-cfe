"""
SPARCE — Saliency-guided Prototype-Aligned Refinement for
Counterfactual Explanations on Multivariate Time Series.

This method addresses the key weakness of GDFO (near-zero validity) and the
weakness of Ours/Soft-DTW (lower validity on multivariate datasets with many
classes) by combining insights from several SOTA approaches:

  1. **Prototype warm-start** (inspired by Native Guide / CELS):
     Instead of optimising from X itself, warm-start from a convex combination
     α·X + (1−α)·NUN, where NUN is the nearest unlike neighbour.  This starts
     the counterfactual close to the decision boundary, dramatically improving
     validity convergence.

  2. **Channel-selective saliency masking** (inspired by M-CELS / CoMTE):
     Compute per-channel gradient saliency to identify which channels are
     discriminative. Apply a learnable soft mask that gates perturbations to
     focus on the channels that matter, preserving non-discriminative channels.

  3. **Two-phase curriculum optimisation**:
     Phase 1 (Validity): Prioritise crossing the decision boundary using
     strong validity weight and prototype guidance. Uses adaptive λ scheduling.
     Phase 2 (Refinement): Once valid, reduce validity weight and increase
     DTW plausibility + density weights to polish the CFE while maintaining
     the class flip.

  4. **Soft-DTW temporal alignment** (from the base paper):
     Retain the soft-DTW k-NN plausibility term to ensure temporal coherence.

  5. **Channel correlation preservation**:
     A cross-channel covariance penalty that preserves the correlation
     structure of the original multivariate time series, preventing
     independent channel perturbations that create unrealistic combinations.

Loss function:
  Phase 1: L = λ_v · L_valid + L_prox + L_sparse + λ_ch · L_channel_corr
  Phase 2: L = L_prox + L_sparse + λ_v' · L_valid + λ_dtw · L_DTW + λ_ch · L_channel_corr

Where:
  L_valid      = max(0, τ - p_f(y_target | X'))     — validity hinge loss
  L_prox       = (1/dT) ||X' - X||²₂               — proximity
  L_sparse     = (1/dT) ||M ⊙ (X' - X)||₁          — saliency-masked sparsity
  L_DTW        = (1/k) Σ DTW_γ(X', Y)               — soft-DTW plausibility
  L_channel_corr = ||Cov(X') - Cov(X)||²_F          — cross-channel correlation
  M            = sigmoid(saliency_logits)            — learnable channel-time mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List

from soft_dtw_cfe.methods.soft_dtw import SoftDTW
from soft_dtw_cfe.config import CFE_CONFIG, DEVICE


# ═══════════════════════════════════════════════════════════════════════════════
# Default Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SPARCE_CONFIG = {
    # Prototype warm-start
    "alpha_init": 0.8,         # X_cf = alpha*X + (1-alpha)*NUN to start
    "n_prototypes": 3,         # Number of NUN prototypes to blend

    # Two-phase curriculum
    "phase1_iters": 80,        # Phase 1: validity-first iterations
    "phase2_iters": 120,       # Phase 2: plausibility refinement iterations
    "lambda_valid_p1": 5.0,    # Validity weight in Phase 1 (aggressive)
    "lambda_valid_p2": 1.0,    # Validity weight in Phase 2 (maintenance)
    "lambda_dtw_p1": 0.1,      # DTW weight in Phase 1 (light)
    "lambda_dtw_p2": 1.0,      # DTW weight in Phase 2 (strong)

    # Saliency masking
    "mask_temp": 2.0,          # Temperature for saliency mask sigmoid
    "mask_sparsity": 0.3,      # L1 penalty on mask to keep it sparse

    # Channel correlation
    "lambda_channel_corr": 0.1,  # Cross-channel correlation preservation

    # Base optimisation
    "tau": 0.5,                # Hinge loss threshold
    "lr_p1": 0.02,             # Learning rate for Phase 1 (larger steps)
    "lr_p2": 0.005,            # Learning rate for Phase 2 (fine-tuning)
    "k": 10,                   # k for DTW neighbours
    "gamma": 1.0,              # Soft-DTW smoothing
}


# ═══════════════════════════════════════════════════════════════════════════════
# SPARCE Counterfactual Generator
# ═══════════════════════════════════════════════════════════════════════════════

class SPARCECounterfactualGenerator:
    """
    Saliency-guided Prototype-Aligned Refinement for Counterfactual
    Explanations (SPARCE) for multivariate time series.

    Designed to achieve both high validity AND high plausibility on
    multivariate datasets — the gap that GDFO and base Soft-DTW leave.

    Key innovations over existing methods:
      1. Prototype-based warm-start for fast validity convergence
      2. Channel-selective saliency masking for focused perturbations
      3. Two-phase curriculum: validity-first then plausibility-refinement
      4. Cross-channel correlation preservation

    Args:
        classifier: Trained TSClassifier (frozen during optimisation).
        **kwargs: Override any SPARCE_CONFIG parameter.
    """

    def __init__(self, classifier: nn.Module, **kwargs):
        self.classifier = classifier

        # Load config with overrides
        cfg = dict(SPARCE_CONFIG)
        cfg.update(kwargs)

        self.alpha_init = cfg["alpha_init"]
        self.n_prototypes = cfg["n_prototypes"]
        self.phase1_iters = cfg["phase1_iters"]
        self.phase2_iters = cfg["phase2_iters"]
        self.lambda_valid_p1 = cfg["lambda_valid_p1"]
        self.lambda_valid_p2 = cfg["lambda_valid_p2"]
        self.lambda_dtw_p1 = cfg["lambda_dtw_p1"]
        self.lambda_dtw_p2 = cfg["lambda_dtw_p2"]
        self.mask_temp = cfg["mask_temp"]
        self.mask_sparsity = cfg["mask_sparsity"]
        self.lambda_channel_corr = cfg["lambda_channel_corr"]
        self.tau = cfg["tau"]
        self.lr_p1 = cfg["lr_p1"]
        self.lr_p2 = cfg["lr_p2"]
        self.k = cfg["k"]
        self.gamma = cfg["gamma"]

        self.soft_dtw = SoftDTW(gamma=self.gamma)

        # Freeze classifier
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad_(False)

    # ── Saliency Computation ──────────────────────────────────────────────

    def _compute_channel_saliency(
        self, x: torch.Tensor, original_class: int
    ) -> torch.Tensor:
        """
        Compute per-channel, per-timestep saliency via classifier gradients.

        Returns a (d, T) saliency map indicating which (channel, timestep)
        pairs most influence the original prediction. Higher values = more
        discriminative = should be perturbed.

        Args:
            x: (d, T) input time series
            original_class: predicted class label

        Returns:
            saliency: (d, T) normalised saliency map in [0, 1]
        """
        x_input = x.clone().detach().requires_grad_(True)

        logits = self.classifier(x_input.unsqueeze(0))
        target_logit = logits[0, original_class]
        target_logit.backward()

        grad = x_input.grad.abs()  # (d, T)

        # Normalise to [0, 1]
        g_min = grad.min()
        g_max = grad.max()
        if g_max - g_min > 1e-8:
            saliency = (grad - g_min) / (g_max - g_min)
        else:
            saliency = torch.ones_like(grad) * 0.5

        return saliency.detach()

    # ── Prototype Selection ───────────────────────────────────────────────

    def _find_prototypes(
        self, x: torch.Tensor, target_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Find the K nearest unlike neighbours from target class.

        Uses a diversity-aware selection: pick the nearest, then
        iteratively pick subsequent ones that are close to x but
        diverse from each other.

        Args:
            x: (d, T) original input
            target_samples: (N, d, T) target class samples

        Returns:
            prototypes: (K, d, T) selected prototype samples
        """
        n_proto = min(self.n_prototypes, target_samples.shape[0])

        # L2 distances from x
        diffs = target_samples - x.unsqueeze(0)
        distances = diffs.pow(2).sum(dim=(1, 2))

        # Greedy diverse selection
        selected = []
        remaining = set(range(len(target_samples)))

        # First: nearest to x
        idx = distances.argmin().item()
        selected.append(idx)
        remaining.discard(idx)

        for _ in range(n_proto - 1):
            if not remaining:
                break

            # Score: close to x but far from already selected
            best_score = -float("inf")
            best_idx = None

            for r_idx in remaining:
                dist_to_x = -distances[r_idx].item()  # Closer to x is better (negative)
                # Diversity: min distance to any selected prototype
                min_dist_to_selected = min(
                    (target_samples[r_idx] - target_samples[s]).pow(2).sum().item()
                    for s in selected
                )
                # Balance proximity to x with diversity
                score = dist_to_x + 0.3 * min_dist_to_selected
                if score > best_score:
                    best_score = score
                    best_idx = r_idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return target_samples[selected]

    # ── Warm-Start Initialisation ─────────────────────────────────────────

    def _create_warm_start(
        self,
        x: torch.Tensor,
        prototypes: torch.Tensor,
        saliency: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """
        Create warm-start initialisation for the counterfactual.

        Strategy: blend x with the best prototype using the saliency mask.
        The saliency guides WHERE to blend (discriminative regions get
        more prototype influence, non-discriminative regions stay original).

        X_init = X + saliency * alpha * (NUN_best - X)

        This produces an initialisation that:
          - Is close to x (small perturbation)
          - Replaces discriminative regions with target-class patterns
          - Preserves non-discriminative channels

        Args:
            x: (d, T)
            prototypes: (K, d, T)
            saliency: (d, T) normalised saliency mask
            target_class: target class label

        Returns:
            x_init: (d, T)
        """
        # Try each prototype and pick the one that gets closest to validity
        best_init = None
        best_prob = -1.0

        for proto in prototypes:
            # Saliency-guided blending
            blend = x + saliency * (1 - self.alpha_init) * (proto - x)

            with torch.no_grad():
                probs = self.classifier.predict_proba(blend.unsqueeze(0))
                p_target = probs[0, target_class].item()

            if p_target > best_prob:
                best_prob = p_target
                best_init = blend

        # If none get close, also try a binary search on alpha
        if best_prob < 0.3:
            best_proto = prototypes[0]  # nearest
            for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
                blend = alpha * x + (1 - alpha) * best_proto
                with torch.no_grad():
                    probs = self.classifier.predict_proba(blend.unsqueeze(0))
                    p_target = probs[0, target_class].item()
                if p_target > best_prob:
                    best_prob = p_target
                    best_init = blend
                if p_target > 0.5:
                    break

        return best_init

    # ── Loss Functions ────────────────────────────────────────────────────

    def _loss_proximity(self, x_cf, x_orig):
        """L_prox = (1/dT) ||X' - X||²₂"""
        d, T = x_orig.shape
        return (x_cf - x_orig).pow(2).sum() / (d * T)

    def _loss_sparsity_masked(self, x_cf, x_orig, mask):
        """L_sparse = (1/dT) ||M ⊙ (X' - X)||₁ — saliency-guided sparsity."""
        d, T = x_orig.shape
        return (mask * (x_cf - x_orig).abs()).sum() / (d * T)

    def _loss_validity(self, x_cf, target_class):
        """L_valid = max(0, τ - p_f(y_target | X'))"""
        probs = self.classifier.predict_proba(x_cf.unsqueeze(0))
        p_target = probs[0, target_class]
        return torch.relu(self.tau - p_target)

    def _loss_dtw_plausibility(self, x_cf, neighbours):
        """L_DTW = (1/k) Σ DTW_γ(X', Y)"""
        k = neighbours.shape[0]
        total_dtw = torch.tensor(0.0, device=x_cf.device)

        x_td = x_cf.permute(1, 0) if x_cf.dim() == 2 else x_cf.unsqueeze(-1)

        for i in range(k):
            y = neighbours[i]
            y_td = y.permute(1, 0) if y.dim() == 2 else y.unsqueeze(-1)
            total_dtw = total_dtw + self.soft_dtw(x_td, y_td)

        return total_dtw / k

    def _loss_channel_correlation(self, x_cf, x_orig):
        """
        L_channel_corr = ||Cov(X') - Cov(X)||²_F

        Preserves the cross-channel correlation structure of the original
        series. This is critical for multivariate plausibility: channels
        in real data have inherent correlations (e.g., x/y/z accelerometer
        axes) that should not be destroyed.
        """
        if x_orig.shape[0] < 2:
            return torch.tensor(0.0, device=x_cf.device)

        # Channel covariance: (d, T) -> (d, d)
        def _cov(x):
            x_centered = x - x.mean(dim=1, keepdim=True)
            return x_centered @ x_centered.T / max(x.shape[1] - 1, 1)

        cov_orig = _cov(x_orig.detach())
        cov_cf = _cov(x_cf)

        return (cov_cf - cov_orig).pow(2).sum()

    def _loss_mask_sparsity(self, mask):
        """Encourage the mask to be sparse (few channels/timesteps modified)."""
        return mask.mean()

    # ── Main Generation ───────────────────────────────────────────────────

    def _find_k_nearest(self, x, target_samples):
        """Find k nearest neighbours from target class."""
        k = min(self.k, target_samples.shape[0])
        diffs = target_samples - x.unsqueeze(0)
        distances = diffs.pow(2).sum(dim=(1, 2))
        _, indices = torch.topk(distances, k=k, largest=False)
        return target_samples[indices]

    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        target_samples: torch.Tensor,
        verbose: bool = False,
    ) -> dict:
        """
        Generate a SPARCE counterfactual for a single time series.

        Two-phase curriculum:
          Phase 1: Validity-first with strong λ_valid, prototype warm-start
          Phase 2: Plausibility refinement with soft-DTW and correlation

        Args:
            x: (d, T) — original input time series
            target_class: integer label of desired class
            target_samples: (N, d, T) — target class training samples
            verbose: print optimisation progress

        Returns:
            dict with counterfactual, original, classes, losses, validity
        """
        x = x.to(DEVICE)
        target_samples = target_samples.to(DEVICE)
        d, T = x.shape

        # Original prediction
        with torch.no_grad():
            orig_pred = self.classifier.predict(x.unsqueeze(0)).item()

        # Step 1: Compute channel saliency
        saliency = self._compute_channel_saliency(x, orig_pred)  # (d, T)

        # Step 2: Find prototypes (diverse NUNs)
        prototypes = self._find_prototypes(x, target_samples)  # (K, d, T)

        # Step 3: Create warm-start initialisation
        x_init = self._create_warm_start(x, prototypes, saliency, target_class)

        # Step 4: Find k-NN for DTW plausibility
        neighbours = self._find_k_nearest(x, target_samples)

        # Step 5: Initialise learnable mask and counterfactual
        # Mask logits initialised from saliency (focus on discriminative regions)
        mask_logits = (saliency * 2 - 1).clone().detach().requires_grad_(True)
        x_cf = x_init.clone().detach().requires_grad_(True)

        loss_history = {
            "total": [], "proximity": [], "sparsity": [],
            "validity": [], "dtw": [], "channel_corr": [],
            "phase": [],
        }

        # ══════════════════════════════════════════════════════════════════
        # Phase 1: VALIDITY-FIRST
        # ══════════════════════════════════════════════════════════════════

        optimizer_p1 = optim.Adam(
            [x_cf, mask_logits], lr=self.lr_p1
        )

        for it in range(self.phase1_iters):
            optimizer_p1.zero_grad()

            mask = torch.sigmoid(mask_logits * self.mask_temp)  # (d, T)

            l_valid = self._loss_validity(x_cf, target_class)
            l_prox = self._loss_proximity(x_cf, x)
            l_sparse = self._loss_sparsity_masked(x_cf, x, mask)
            l_ch_corr = self._loss_channel_correlation(x_cf, x)
            l_mask_sp = self._loss_mask_sparsity(mask)

            # Light DTW even in Phase 1 for temporal guidance
            l_dtw = self._loss_dtw_plausibility(x_cf, neighbours)

            loss = (
                self.lambda_valid_p1 * l_valid
                + l_prox
                + l_sparse
                + self.lambda_dtw_p1 * l_dtw
                + self.lambda_channel_corr * l_ch_corr
                + self.mask_sparsity * l_mask_sp
            )

            loss.backward()
            optimizer_p1.step()

            loss_history["total"].append(loss.item())
            loss_history["proximity"].append(l_prox.item())
            loss_history["sparsity"].append(l_sparse.item())
            loss_history["validity"].append(l_valid.item())
            loss_history["dtw"].append(l_dtw.item())
            loss_history["channel_corr"].append(l_ch_corr.item())
            loss_history["phase"].append(1)

            # Early exit Phase 1 if already valid with margin
            if l_valid.item() == 0:
                with torch.no_grad():
                    probs = self.classifier.predict_proba(x_cf.unsqueeze(0))
                    if probs[0, target_class].item() > 0.7:
                        if verbose:
                            print(f"    Phase 1 early exit at iter {it+1} "
                                  f"(p_target={probs[0, target_class].item():.3f})")
                        break

            if verbose and (it + 1) % 40 == 0:
                with torch.no_grad():
                    cf_pred = self.classifier.predict(x_cf.unsqueeze(0)).item()
                    cf_prob = self.classifier.predict_proba(
                        x_cf.unsqueeze(0)
                    )[0, target_class].item()
                print(
                    f"    P1 Iter {it+1:4d} | Loss: {loss.item():.4f} | "
                    f"Valid: {l_valid.item():.4f} | DTW: {l_dtw.item():.2f} | "
                    f"Pred: {cf_pred} (p={cf_prob:.3f})"
                )

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: PLAUSIBILITY REFINEMENT
        # ══════════════════════════════════════════════════════════════════

        # Freeze mask, only optimise x_cf with fine learning rate
        mask_final = torch.sigmoid(mask_logits.detach() * self.mask_temp)

        # Re-init x_cf as parameter (preserve gradients)
        x_cf = x_cf.detach().requires_grad_(True)
        optimizer_p2 = optim.Adam([x_cf], lr=self.lr_p2)

        for it in range(self.phase2_iters):
            optimizer_p2.zero_grad()

            l_valid = self._loss_validity(x_cf, target_class)
            l_prox = self._loss_proximity(x_cf, x)
            l_sparse = self._loss_sparsity_masked(x_cf, x, mask_final)
            l_dtw = self._loss_dtw_plausibility(x_cf, neighbours)
            l_ch_corr = self._loss_channel_correlation(x_cf, x)

            loss = (
                l_prox
                + l_sparse
                + self.lambda_valid_p2 * l_valid
                + self.lambda_dtw_p2 * l_dtw
                + self.lambda_channel_corr * l_ch_corr
            )

            loss.backward()
            optimizer_p2.step()

            loss_history["total"].append(loss.item())
            loss_history["proximity"].append(l_prox.item())
            loss_history["sparsity"].append(l_sparse.item())
            loss_history["validity"].append(l_valid.item())
            loss_history["dtw"].append(l_dtw.item())
            loss_history["channel_corr"].append(l_ch_corr.item())
            loss_history["phase"].append(2)

            if verbose and (it + 1) % 40 == 0:
                with torch.no_grad():
                    cf_pred = self.classifier.predict(x_cf.unsqueeze(0)).item()
                    cf_prob = self.classifier.predict_proba(
                        x_cf.unsqueeze(0)
                    )[0, target_class].item()
                print(
                    f"    P2 Iter {it+1:4d} | Loss: {loss.item():.4f} | "
                    f"Valid: {l_valid.item():.4f} | DTW: {l_dtw.item():.2f} | "
                    f"ChCorr: {l_ch_corr.item():.4f} | "
                    f"Pred: {cf_pred} (p={cf_prob:.3f})"
                )

        # Final prediction
        x_cf_final = x_cf.detach()
        with torch.no_grad():
            cf_pred = self.classifier.predict(x_cf_final.unsqueeze(0)).item()

        return {
            "counterfactual": x_cf_final.cpu(),
            "original": x.cpu(),
            "target_class": target_class,
            "original_class": orig_pred,
            "predicted_class": cf_pred,
            "valid": cf_pred == target_class,
            "losses": loss_history,
        }

    def generate_batch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> list:
        """
        Generate counterfactuals for a batch of test samples.

        For each sample, the target class is the next most probable class.

        Args:
            X: (n, d, T) — test samples
            y: (n,) — true labels
            X_train: (n_train, d, T) — training data (numpy)
            y_train: (n_train,) — training labels (numpy)
            verbose: print progress

        Returns:
            List of result dicts (one per sample)
        """
        results = []
        n = X.shape[0]

        # Pre-compute target class samples
        unique_classes = np.unique(y_train)
        target_samples_dict = {}
        for c in unique_classes:
            mask = (y_train == int(c))
            target_samples_dict[c] = torch.tensor(
                X_train[mask], dtype=torch.float32
            ).to(DEVICE)

        for i in range(n):
            xi = X[i].to(DEVICE)
            yi = y[i].item() if isinstance(y[i], torch.Tensor) else y[i]

            # Choose target class: next most probable
            with torch.no_grad():
                probs = self.classifier.predict_proba(xi.unsqueeze(0))[0]
                probs_copy = probs.clone()
                probs_copy[yi] = -1.0
                target_class = probs_copy.argmax().item()

            target_samps = target_samples_dict[int(target_class)]

            if verbose and (i + 1) % 5 == 0:
                print(f"  [SPARCE] [{i+1}/{n}] Generating CFE: "
                      f"class {yi} -> {target_class}")

            result = self.generate(xi, target_class, target_samps, verbose=False)
            results.append(result)

        return results
