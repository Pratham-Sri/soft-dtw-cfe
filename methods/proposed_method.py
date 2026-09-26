"""
Proposed Method: Soft-DTW Counterfactual Explanations.

Complete implementation of Section 4 — "Method":

  L_CF = L_prox + L_sparse + λ · (L_valid + L_DTW)      (Equation 3)

Where:
  L_prox   = (1/dT) ||X' - X||²₂           — proximity (squared Euclidean)
  L_sparse = (1/dT) ||X' - X||₁            — sparsity (L1 norm)
  L_valid  = max(0, τ - p_f(y_target|X'))   — validity (hinge loss)
  L_DTW    = (1/k) Σ_{Y∈N_k} DTW_γ(X', Y)  — plausibility (soft-DTW to k-NN)

Optimisation: gradient descent on X' for fixed number of iterations,
with classifier parameters held constant.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from soft_dtw_cfe.methods.soft_dtw import SoftDTW, SoftDTWBatch
from soft_dtw_cfe.config import CFE_CONFIG, DEVICE


class SoftDTWCounterfactualGenerator:
    """
    Generates plausible counterfactual explanations for time series
    via gradient-based optimisation with soft-DTW alignment.
    
    This is the main proposed method from the paper. The counterfactual
    X' is optimised directly in input space while the classifier is frozen.
    
    Args:
        classifier: Trained TSClassifier (frozen during optimisation).
        gamma: Soft-DTW smoothing parameter γ (default 1.0).
        lambda_: Balance parameter λ for validity+plausibility vs prox+sparse.
        k: Number of nearest target-class neighbours for DTW alignment.
        tau: Hinge loss threshold τ for validity.
        lr: Learning rate for Adam optimiser.
        num_iterations: Number of gradient descent steps.
    """

    def __init__(
        self,
        classifier: nn.Module,
        gamma: float = None,
        lambda_: float = None,
        k: int = None,
        tau: float = None,
        lr: float = None,
        num_iterations: int = None,
    ):
        self.classifier = classifier
        self.gamma = gamma if gamma is not None else CFE_CONFIG["gamma"]
        self.lambda_ = lambda_ if lambda_ is not None else CFE_CONFIG["lambda_"]
        self.k = k if k is not None else CFE_CONFIG["k"]
        self.tau = tau if tau is not None else CFE_CONFIG["tau"]
        self.lr = lr if lr is not None else CFE_CONFIG["lr"]
        self.num_iterations = num_iterations if num_iterations is not None else CFE_CONFIG["num_iterations"]
        
        self.soft_dtw = SoftDTW(gamma=self.gamma)

        # Freeze classifier
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad_(False)

    def _find_k_nearest_neighbours(
        self,
        x: torch.Tensor,
        target_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find k nearest neighbours of x from target_samples using L2 distance.
        
        N_k(X, y_target) in the paper.
        
        Args:
            x: (d, T) — original input
            target_samples: (N, d, T) — all target class samples
        
        Returns:
            neighbours: (k, d, T)
        """
        k = min(self.k, target_samples.shape[0])
        
        # L2 distances
        diffs = target_samples - x.unsqueeze(0)  # (N, d, T)
        distances = diffs.pow(2).sum(dim=(1, 2))  # (N,)
        
        _, indices = torch.topk(distances, k=k, largest=False)
        return target_samples[indices]

    def _loss_proximity(self, x_cf: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """
        Proximity loss: L_prox = (1/dT) ||X' - X||²₂
        
        Enforces similarity between counterfactual and original input,
        ensuring minimal perturbation.
        """
        d = x_orig.shape[0]
        T = x_orig.shape[1]
        return (x_cf - x_orig).pow(2).sum() / (d * T)

    def _loss_sparsity(self, x_cf: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """
        Sparsity loss: L_sparse = (1/dT) ||X' - X||₁
        
        Encourages localized modifications by penalizing the L1 norm,
        promoting concentration of modifications in specific regions.
        """
        d = x_orig.shape[0]
        T = x_orig.shape[1]
        return (x_cf - x_orig).abs().sum() / (d * T)

    def _loss_validity(self, x_cf: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Validity loss: L_valid = max(0, τ - p_f(y_target|X'))
        
        Hinge loss ensuring the classifier assigns X' to the target class
        with confidence at least τ.
        """
        probs = self.classifier.predict_proba(x_cf.unsqueeze(0))  # (1, c)
        p_target = probs[0, target_class]
        return torch.relu(self.tau - p_target)

    def _loss_dtw_plausibility(
        self, x_cf: torch.Tensor, neighbours: torch.Tensor
    ) -> torch.Tensor:
        """
        Plausibility loss: L_DTW = (1/k) Σ_{Y∈N_k} DTW_γ(X', Y)
        
        Aligns counterfactual with real target samples via soft-DTW,
        ensuring realistic temporal patterns.
        """
        k = neighbours.shape[0]
        total_dtw = torch.tensor(0.0, device=x_cf.device)
        
        # x_cf: (d, T) → (T, d) for soft-DTW
        x_td = x_cf.permute(1, 0) if x_cf.dim() == 2 else x_cf.unsqueeze(-1)
        
        for i in range(k):
            y = neighbours[i]  # (d, T)
            y_td = y.permute(1, 0) if y.dim() == 2 else y.unsqueeze(-1)  # (T, d)
            total_dtw = total_dtw + self.soft_dtw(x_td, y_td)
        
        return total_dtw / k

    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        target_samples: torch.Tensor,
        verbose: bool = False,
    ) -> dict:
        """
        Generate a counterfactual explanation for a single time series.
        
        Implements Algorithm from Section 4 (Equation 3):
          L_CF = L_prox + L_sparse + λ · (L_valid + L_DTW)
        
        Args:
            x: (d, T) — original input time series.
            target_class: Integer label of the desired target class.
            target_samples: (N, d, T) — all training samples from target class.
            verbose: Print optimisation progress.
        
        Returns:
            dict with:
                'counterfactual': (d, T) tensor — the generated X'
                'original': (d, T) tensor — the original X
                'target_class': int
                'original_class': int
                'losses': dict of loss component histories
                'valid': bool — whether f(X') == target_class
        """
        x = x.to(DEVICE)
        target_samples = target_samples.to(DEVICE)

        # Get original prediction
        with torch.no_grad():
            orig_pred = self.classifier.predict(x.unsqueeze(0)).item()

        # Find k nearest neighbours from target class
        neighbours = self._find_k_nearest_neighbours(x, target_samples)

        # Initialise counterfactual as a copy of the original
        x_cf = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_cf], lr=self.lr)

        loss_history = {
            "total": [], "proximity": [], "sparsity": [],
            "validity": [], "dtw": [],
        }

        for it in range(self.num_iterations):
            optimizer.zero_grad()

            # Compute all loss components
            l_prox = self._loss_proximity(x_cf, x)
            l_sparse = self._loss_sparsity(x_cf, x)
            l_valid = self._loss_validity(x_cf, target_class)
            l_dtw = self._loss_dtw_plausibility(x_cf, neighbours)

            # Total loss: Equation (3)
            # L_CF = L_prox + L_sparse + λ · (L_valid + L_DTW)
            loss = l_prox + l_sparse + self.lambda_ * (l_valid + l_dtw)

            loss.backward()
            optimizer.step()

            # Record
            loss_history["total"].append(loss.item())
            loss_history["proximity"].append(l_prox.item())
            loss_history["sparsity"].append(l_sparse.item())
            loss_history["validity"].append(l_valid.item())
            loss_history["dtw"].append(l_dtw.item())

            if verbose and (it + 1) % 100 == 0:
                with torch.no_grad():
                    cf_pred = self.classifier.predict(x_cf.unsqueeze(0)).item()
                    cf_prob = self.classifier.predict_proba(x_cf.unsqueeze(0))[0, target_class].item()
                print(
                    f"  Iter {it+1:4d} | Loss: {loss.item():.4f} | "
                    f"Prox: {l_prox.item():.4f} | Sparse: {l_sparse.item():.4f} | "
                    f"Valid: {l_valid.item():.4f} | DTW: {l_dtw.item():.4f} | "
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
        
        For each sample, the target class is chosen as the class
        with the second-highest predicted probability (nearest
        alternative class).
        
        Args:
            X: (n, d, T) — test samples.
            y: (n,) — true labels.
            X_train: (n_train, d, T) — training data (numpy).
            y_train: (n_train,) — training labels (numpy).
            verbose: Print progress.
        
        Returns:
            List of result dicts (one per sample).
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
            xi = X[i].to(DEVICE)  # (d, T)
            yi = y[i].item() if isinstance(y[i], torch.Tensor) else y[i]

            # Choose target class: next most probable class ≠ original
            with torch.no_grad():
                probs = self.classifier.predict_proba(xi.unsqueeze(0))[0]  # (c,)
                probs_copy = probs.clone()
                probs_copy[yi] = -1.0
                target_class = probs_copy.argmax().item()

            # Get target class training samples
            target_samps = target_samples_dict[int(target_class)]

            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n}] Generating CFE: class {yi} -> {target_class}")

            result = self.generate(xi, target_class, target_samps, verbose=False)
            results.append(result)

        return results
