"""
M-CELS: Counterfactual Explanations for Multivariate Time Series
Data Guided by Learned Saliency Maps.

Reference: Li et al. (2024) — Reference [12] in the paper.

M-CELS learns differentiable saliency maps to identify the most
influential time steps for classification decisions. This map is
then used to guide sparse perturbations by replacing salient time
steps with values from a nearest unlike neighbour (NUN).

Key idea:
  1. Compute a saliency map via gradient-based attribution
  2. Find a Nearest Unlike Neighbour (NUN) from the target class
  3. Replace salient regions of the original with NUN values
  4. Iteratively refine until the classifier flips
"""

import torch
import torch.nn as nn
import numpy as np
import copy

from soft_dtw_cfe.config import DEVICE


class MCELS:
    """
    M-CELS counterfactual generation.
    
    Uses learned saliency maps to identify important time steps
    and replaces them with target-class patterns.
    
    Args:
        classifier: Trained classifier (frozen).
        max_iterations: Maximum refinement iterations.
        top_k_ratio: Fraction of most salient time steps to replace.
    """

    def __init__(
        self,
        classifier: nn.Module,
        max_iterations: int = 50,
        top_k_ratio: float = 0.3,
    ):
        self.classifier = classifier
        self.max_iterations = max_iterations
        self.top_k_ratio = top_k_ratio

        # Freeze classifier
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad_(False)

    def _compute_saliency(self, x: torch.Tensor, original_class: int) -> torch.Tensor:
        """
        Compute gradient-based saliency map.
        
        The saliency at each time step indicates how much that
        step contributes to the original class prediction.
        
        Args:
            x: (d, T) — input time series
            original_class: predicted class label
        
        Returns:
            saliency: (T,) — importance of each time step
        """
        x_input = x.clone().detach().requires_grad_(True)
        
        logits = self.classifier(x_input.unsqueeze(0))  # (1, c)
        target_logit = logits[0, original_class]
        target_logit.backward()
        
        # Aggregate gradient magnitude across channels
        grad = x_input.grad  # (d, T)
        saliency = grad.abs().sum(dim=0)  # (T,)
        
        return saliency.detach()

    def _find_nearest_unlike_neighbour(
        self,
        x: torch.Tensor,
        target_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find the Nearest Unlike Neighbour (NUN) — the closest
        training sample from the target class.
        
        Args:
            x: (d, T)
            target_samples: (N, d, T)
        
        Returns:
            nun: (d, T) — nearest target-class sample
        """
        diffs = target_samples - x.unsqueeze(0)
        distances = diffs.pow(2).sum(dim=(1, 2))
        idx = distances.argmin()
        return target_samples[idx]

    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        target_samples: torch.Tensor,
        verbose: bool = False,
    ) -> dict:
        """
        Generate an M-CELS counterfactual.
        
        Process:
          1. Compute saliency map for original prediction
          2. Find NUN from target class
          3. Iteratively replace most salient time steps with NUN values
          4. Stop when classifier flips or max iterations reached
        
        Args:
            x: (d, T) — original time series
            target_class: desired target class
            target_samples: (N, d, T) — target class training samples
        
        Returns:
            dict with counterfactual results
        """
        x = x.to(DEVICE)
        target_samples = target_samples.to(DEVICE)

        with torch.no_grad():
            orig_pred = self.classifier.predict(x.unsqueeze(0)).item()

        # Find NUN
        nun = self._find_nearest_unlike_neighbour(x, target_samples)  # (d, T)

        # Compute saliency
        saliency = self._compute_saliency(x, orig_pred)  # (T,)

        T = x.shape[-1]
        top_k = max(1, int(T * self.top_k_ratio))

        # Sort time steps by saliency (descending)
        _, sorted_indices = torch.sort(saliency, descending=True)

        x_cf = x.clone()
        valid = False

        for it in range(self.max_iterations):
            # Determine how many time steps to replace this iteration
            n_replace = min(top_k * (it + 1), T)
            indices_to_replace = sorted_indices[:n_replace]

            # Replace salient time steps with NUN values
            x_cf_candidate = x.clone()
            x_cf_candidate[:, indices_to_replace] = nun[:, indices_to_replace]

            with torch.no_grad():
                pred = self.classifier.predict(x_cf_candidate.unsqueeze(0)).item()

            if pred == target_class:
                x_cf = x_cf_candidate
                valid = True
                break

            x_cf = x_cf_candidate

        with torch.no_grad():
            cf_pred = self.classifier.predict(x_cf.unsqueeze(0)).item()

        return {
            "counterfactual": x_cf.cpu(),
            "original": x.cpu(),
            "target_class": target_class,
            "original_class": orig_pred,
            "predicted_class": cf_pred,
            "valid": cf_pred == target_class,
        }

    def generate_batch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> list:
        """Generate counterfactuals for a batch of test samples."""
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

            # Choose target class
            with torch.no_grad():
                probs = self.classifier.predict_proba(xi.unsqueeze(0))[0]
                probs_copy = probs.clone()
                probs_copy[yi] = -1.0
                target_class = probs_copy.argmax().item()

            target_samps = target_samples_dict[int(target_class)]

            if verbose and (i + 1) % 20 == 0:
                print(f"  [M-CELS] [{i+1}/{n}] class {yi} -> {target_class}")

            result = self.generate(xi, target_class, target_samps, verbose=False)
            results.append(result)

        return results
