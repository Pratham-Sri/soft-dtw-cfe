"""
Generative Density Function Optimisation (GDFO) for Time Series
Counterfactual Explanations.

This method introduces a density-aware approach to counterfactual generation
for multivariate time series. Instead of relying solely on k-nearest-neighbour
alignment (as in the Soft-DTW method) or latent-space perturbation (Glacier),
GDFO:

  1. Learns a class-conditional density model via a Variational Autoencoder
     (VAE) with a Gaussian Mixture Model (GMM) prior in latent space.
  2. Generates counterfactuals by optimising in input space to:
     (a) Flip the classifier's prediction to the target class (validity),
     (b) Maximise the learned density under the target class (plausibility),
     (c) Remain close to the original input (proximity / sparsity).

Loss function:
  L_GDFO = L_prox + L_sparse + λ_v · L_valid + λ_d · L_density + λ_dtw · L_dtw

Where:
  L_prox    = (1/dT) ||X' - X||²₂
  L_sparse  = (1/dT) ||X' - X||₁
  L_valid   = max(0, τ - p_f(y_target | X'))
  L_density = -log p_θ(X' | y_target)   (negative log-density under target class)
  L_dtw     = (1/k) Σ DTW_γ(X', Y)      (optional soft-DTW alignment)

The density term L_density is the key contribution: it uses the trained VAE
encoder to project X' into latent space, then evaluates the class-conditional
GMM log-probability. This provides a smooth, differentiable signal that guides
counterfactuals towards high-density regions of the target class manifold.

Advantages over pure soft-DTW approach:
  - Captures the full multivariate distribution, not just k nearest neighbours
  - Scales better to larger datasets (density model is trained once, then reused)
  - Naturally handles multivariate time series without channel-independence
  - The GMM prior captures multi-modal class distributions
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
# VAE Architecture — learns a smooth latent manifold for time series
# ═══════════════════════════════════════════════════════════════════════════════

class TSVariationalAutoencoder(nn.Module):
    """
    Convolutional VAE for multivariate time series.

    Architecture:
      Encoder: d → 64 → 128 → 256 → flatten → (μ, log σ²) of dim z_dim
      Decoder: z_dim → unflatten → 256 → 128 → 64 → d

    The encoder produces a latent code z ~ N(μ, σ²I) via the reparameterisation
    trick, enabling gradient-based optimisation through the sampling step.
    """

    def __init__(self, n_channels: int, seq_len: int, z_dim: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.z_dim = z_dim

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        # Compute the flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, seq_len)
            enc_out = self.encoder(dummy)
            self._enc_shape = enc_out.shape[1:]  # (256, T')
            self._flat_dim = enc_out.numel()

        self.fc_mu = nn.Linear(self._flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, z_dim)

        # ── Decoder ──────────────────────────────────────────────────────
        self.fc_decode = nn.Linear(z_dim, self._flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1,
                               output_padding=0),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, n_channels, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
        )

    def encode(self, x: torch.Tensor):
        """Encode x -> (mu, logvar)."""
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterisation trick: z = μ + σ · ε, ε ~ N(0,I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor):
        """Decode z -> x_reconstructed."""
        h = self.fc_decode(z)
        h = h.view(-1, *self._enc_shape)
        x_hat = self.decoder(h)
        # Ensure output matches input seq_len (handle rounding from strided convs)
        if x_hat.shape[-1] != self.seq_len:
            x_hat = F.interpolate(x_hat, size=self.seq_len, mode='linear',
                                  align_corners=False)
        return x_hat

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    def get_latent(self, x: torch.Tensor):
        """Get the deterministic latent embedding (mean only)."""
        mu, _ = self.encode(x)
        return mu


# ═══════════════════════════════════════════════════════════════════════════════
# Class-Conditional Gaussian Mixture Density Model
# ═══════════════════════════════════════════════════════════════════════════════

class ClassConditionalGMM:
    """
    Fits a class-conditional Gaussian Mixture Model in the VAE latent space.

    For each class c, the density p(z | y=c) is modelled as a mixture of
    K Gaussians with full covariance. This captures multi-modal distributions
    that naturally arise in time series (e.g., different gesture sub-types
    within a class).

    The log-density is differentiable w.r.t. z, enabling gradient-based
    optimisation of counterfactuals.
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.class_models = {}  # {class_label: fitted sklearn GMM}

    def fit(self, z_all: np.ndarray, y_all: np.ndarray):
        """
        Fit a GMM per class.

        Args:
            z_all: (N, z_dim) latent embeddings of all training samples.
            y_all: (N,) class labels.
        """
        from sklearn.mixture import GaussianMixture

        for c in np.unique(y_all):
            mask = y_all == c
            z_c = z_all[mask]
            n_c = len(z_c)

            # Adapt number of components to available samples
            k = min(self.n_components, max(1, n_c // 3))

            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                max_iter=200,
                random_state=42,
                reg_covar=1e-5,
            )
            gmm.fit(z_c)
            self.class_models[int(c)] = gmm

    def log_prob_torch(self, z: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Compute differentiable log p(z | y=target_class) using PyTorch ops.

        Manually implements the GMM log-probability with the fitted sklearn
        parameters, so gradients can flow through z.

        Args:
            z: (z_dim,) or (1, z_dim) latent vector (requires_grad=True OK)
            target_class: class label

        Returns:
            scalar log-probability (differentiable w.r.t. z)
        """
        gmm = self.class_models[target_class]
        z_2d = z.unsqueeze(0) if z.dim() == 1 else z  # (1, D)

        # Extract fitted GMM parameters
        weights = torch.tensor(
            gmm.weights_, dtype=torch.float32, device=z.device
        )  # (K,)
        means = torch.tensor(
            gmm.means_, dtype=torch.float32, device=z.device
        )  # (K, D)
        covs = torch.tensor(
            gmm.covariances_, dtype=torch.float32, device=z.device
        )  # (K, D, D)

        K = weights.shape[0]
        D = means.shape[1]

        log_probs = []
        for k in range(K):
            diff = z_2d - means[k].unsqueeze(0)  # (1, D)
            cov_k = covs[k]  # (D, D)

            # Cholesky decomposition for numerical stability
            try:
                L = torch.linalg.cholesky(cov_k)
            except RuntimeError:
                # Add jitter if not positive definite
                cov_k = cov_k + 1e-4 * torch.eye(D, device=z.device)
                L = torch.linalg.cholesky(cov_k)

            # log |Σ| = 2 * sum(log(diag(L)))
            log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))

            # Mahalanobis: (x-μ)^T Σ^{-1} (x-μ) via solve
            solved = torch.linalg.solve_triangular(L, diff.T, upper=False)  # (D, 1)
            mahal = torch.sum(solved ** 2)

            log_p_k = -0.5 * (D * np.log(2 * np.pi) + log_det + mahal)
            log_probs.append(torch.log(weights[k]) + log_p_k)

        log_probs = torch.stack(log_probs)
        return torch.logsumexp(log_probs, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# VAE Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_vae(
    vae: TSVariationalAutoencoder,
    X_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    beta: float = 1.0,
    verbose: bool = True,
) -> TSVariationalAutoencoder:
    """
    Train the VAE on training data with β-VAE objective.

    Loss = Reconstruction + β · KL divergence

    Args:
        vae: TSVariationalAutoencoder instance.
        X_train: (N, d, T) training time series.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        beta: KL weight (β-VAE). Lower values = less regularised latent space.
        verbose: Print training progress.
    """
    vae = vae.to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    n = len(X_tensor)

    vae.train()
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch = X_tensor[perm[i:i + batch_size]]
            optimizer.zero_grad()

            x_hat, mu, logvar = vae(batch)

            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(x_hat, batch, reduction='sum') / len(batch)

            # KL divergence: -0.5 * Σ(1 + logvar - μ² - exp(logvar))
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ) / len(batch)

            loss = recon_loss + beta * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if verbose and (epoch + 1) % 20 == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"    [VAE] Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg:.4f}")

    vae.eval()
    return vae


# ═══════════════════════════════════════════════════════════════════════════════
# GDFO Counterfactual Generator
# ═══════════════════════════════════════════════════════════════════════════════

# Default GDFO hyperparameters
GDFO_CONFIG = {
    "z_dim": 32,            # VAE latent dimension
    "vae_epochs": 80,       # VAE training epochs
    "vae_beta": 0.5,        # β for β-VAE (lower = sharper reconstructions)
    "gmm_components": 5,    # GMM components per class
    "lambda_valid": 2.0,    # validity weight
    "lambda_density": 0.5,  # density weight
    "lambda_dtw": 0.3,      # DTW weight (optional soft-DTW alignment)
    "tau": 0.5,             # hinge loss threshold
    "lr": 0.01,             # optimisation learning rate
    "num_iterations": 200,  # gradient descent steps
    "k": 5,                 # k for DTW neighbours (fewer than base method for speed)
    "gamma": 1.0,           # soft-DTW smoothing
    "use_dtw": True,        # whether to include DTW loss
}


class GDFOCounterfactualGenerator:
    """
    Generative Density Function Optimisation (GDFO) for time series CFEs.

    This method combines:
      1. A class-conditional density model (VAE + GMM) for plausibility
      2. Classifier-guided validity optimisation
      3. Proximity and sparsity constraints
      4. Optional soft-DTW alignment for temporal coherence

    The key insight is that modelling the full density p(x | y_target) provides
    a richer, more informative gradient signal than nearest-neighbour DTW alone.
    The density function captures:
      - Multi-modal class structure (via GMM)
      - Non-linear manifold geometry (via VAE)
      - Cross-channel correlations (via joint encoding)

    This makes it particularly effective for larger multivariate datasets
    where the target class distribution is complex and multi-modal.

    Args:
        classifier: Trained TSClassifier (frozen during optimisation).
        n_channels: Number of input channels (d).
        seq_len: Sequence length (T).
        z_dim: VAE latent dimensionality.
        vae_epochs: VAE training epochs.
        vae_beta: β-VAE KL weight.
        gmm_components: Number of GMM components per class.
        lambda_valid: Weight for validity loss.
        lambda_density: Weight for density loss.
        lambda_dtw: Weight for DTW loss.
        tau: Hinge loss threshold.
        lr: Optimisation learning rate.
        num_iterations: Number of gradient descent steps.
        k: Number of neighbours for DTW alignment.
        gamma: Soft-DTW smoothing parameter.
        use_dtw: Whether to include DTW alignment loss.
    """

    def __init__(
        self,
        classifier: nn.Module,
        n_channels: int,
        seq_len: int,
        z_dim: int = None,
        vae_epochs: int = None,
        vae_beta: float = None,
        gmm_components: int = None,
        lambda_valid: float = None,
        lambda_density: float = None,
        lambda_dtw: float = None,
        tau: float = None,
        lr: float = None,
        num_iterations: int = None,
        k: int = None,
        gamma: float = None,
        use_dtw: bool = None,
    ):
        self.classifier = classifier
        self.n_channels = n_channels
        self.seq_len = seq_len

        # Load defaults from config
        self.z_dim = z_dim or GDFO_CONFIG["z_dim"]
        self.vae_epochs = vae_epochs or GDFO_CONFIG["vae_epochs"]
        self.vae_beta = vae_beta if vae_beta is not None else GDFO_CONFIG["vae_beta"]
        self.gmm_components = gmm_components or GDFO_CONFIG["gmm_components"]
        self.lambda_valid = lambda_valid if lambda_valid is not None else GDFO_CONFIG["lambda_valid"]
        self.lambda_density = lambda_density if lambda_density is not None else GDFO_CONFIG["lambda_density"]
        self.lambda_dtw = lambda_dtw if lambda_dtw is not None else GDFO_CONFIG["lambda_dtw"]
        self.tau = tau if tau is not None else GDFO_CONFIG["tau"]
        self.lr = lr if lr is not None else GDFO_CONFIG["lr"]
        self.num_iterations = num_iterations or GDFO_CONFIG["num_iterations"]
        self.k = k or GDFO_CONFIG["k"]
        self.gamma = gamma if gamma is not None else GDFO_CONFIG["gamma"]
        self.use_dtw = use_dtw if use_dtw is not None else GDFO_CONFIG["use_dtw"]

        # Density model components (trained lazily)
        self.vae = None
        self.gmm = None

        # DTW module
        self.soft_dtw = SoftDTW(gamma=self.gamma)

        # Freeze classifier
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad_(False)

    def _train_density_model(
        self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True
    ):
        """
        Train the VAE + GMM density model on the full training set.

        Steps:
          1. Train VAE on all X_train (class-agnostic reconstruction).
          2. Encode all X_train into latent space z.
          3. Fit a class-conditional GMM on z per class.
        """
        if verbose:
            print("    [GDFO] Training VAE density model...")

        # 1. Train VAE
        self.vae = TSVariationalAutoencoder(
            n_channels=self.n_channels,
            seq_len=self.seq_len,
            z_dim=self.z_dim,
        )
        self.vae = train_vae(
            self.vae, X_train,
            epochs=self.vae_epochs,
            beta=self.vae_beta,
            verbose=verbose,
        )

        # 2. Encode all training data
        self.vae.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
            # Process in batches to avoid OOM on large datasets
            z_list = []
            bs = 128
            for i in range(0, len(X_t), bs):
                batch = X_t[i:i + bs]
                z_batch = self.vae.get_latent(batch)
                z_list.append(z_batch.cpu().numpy())
            z_all = np.concatenate(z_list, axis=0)

        # 3. Fit class-conditional GMM
        if verbose:
            print("    [GDFO] Fitting class-conditional GMM in latent space...")
        self.gmm = ClassConditionalGMM(n_components=self.gmm_components)
        self.gmm.fit(z_all, y_train)

        if verbose:
            for c in sorted(self.gmm.class_models.keys()):
                n_c = np.sum(y_train == c)
                k_c = self.gmm.class_models[c].n_components
                print(f"      Class {c}: {n_c} samples, {k_c} GMM components")

    def _loss_proximity(self, x_cf: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """L_prox = (1/dT) ||X' - X||²₂"""
        d, T = x_orig.shape
        return (x_cf - x_orig).pow(2).sum() / (d * T)

    def _loss_sparsity(self, x_cf: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """L_sparse = (1/dT) ||X' - X||₁"""
        d, T = x_orig.shape
        return (x_cf - x_orig).abs().sum() / (d * T)

    def _loss_validity(self, x_cf: torch.Tensor, target_class: int) -> torch.Tensor:
        """L_valid = max(0, τ - p_f(y_target | X'))"""
        probs = self.classifier.predict_proba(x_cf.unsqueeze(0))
        p_target = probs[0, target_class]
        return torch.relu(self.tau - p_target)

    def _loss_density(self, x_cf: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        L_density = -log p_θ(z' | y_target)

        Projects the counterfactual into VAE latent space and evaluates
        the class-conditional GMM log-probability. Returns negative
        log-density (to be minimised).
        """
        # Encode x_cf through VAE (deterministic: use mean only)
        mu, _ = self.vae.encode(x_cf.unsqueeze(0))
        z = mu.squeeze(0)  # (z_dim,)

        # GMM log-probability under target class
        log_p = self.gmm.log_prob_torch(z, target_class)

        # Return negative log-density (lower = higher density = better)
        return -log_p

    def _loss_dtw_plausibility(
        self, x_cf: torch.Tensor, neighbours: torch.Tensor
    ) -> torch.Tensor:
        """L_DTW = (1/k) Σ DTW_γ(X', Y)"""
        k = neighbours.shape[0]
        total_dtw = torch.tensor(0.0, device=x_cf.device)

        x_td = x_cf.permute(1, 0) if x_cf.dim() == 2 else x_cf.unsqueeze(-1)

        for i in range(k):
            y = neighbours[i]
            y_td = y.permute(1, 0) if y.dim() == 2 else y.unsqueeze(-1)
            total_dtw = total_dtw + self.soft_dtw(x_td, y_td)

        return total_dtw / k

    def _find_k_nearest(
        self, x: torch.Tensor, target_samples: torch.Tensor
    ) -> torch.Tensor:
        """Find k nearest neighbours of x from target_samples using L2."""
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
        Generate a counterfactual explanation for a single time series.

        L_GDFO = L_prox + L_sparse + λ_v · L_valid + λ_d · L_density + λ_dtw · L_dtw

        Args:
            x: (d, T) — original input time series.
            target_class: Integer label of the desired target class.
            target_samples: (N, d, T) — all training samples from target class.
            verbose: Print optimisation progress.

        Returns:
            dict with counterfactual, original, classes, losses, validity.
        """
        x = x.to(DEVICE)
        target_samples = target_samples.to(DEVICE)

        # Get original prediction
        with torch.no_grad():
            orig_pred = self.classifier.predict(x.unsqueeze(0)).item()

        # Find k nearest neighbours for DTW (if enabled)
        neighbours = None
        if self.use_dtw:
            neighbours = self._find_k_nearest(x, target_samples)

        # Initialise counterfactual as copy of original
        x_cf = x.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([x_cf], lr=self.lr)

        loss_history = {
            "total": [], "proximity": [], "sparsity": [],
            "validity": [], "density": [], "dtw": [],
        }

        for it in range(self.num_iterations):
            optimizer.zero_grad()

            l_prox = self._loss_proximity(x_cf, x)
            l_sparse = self._loss_sparsity(x_cf, x)
            l_valid = self._loss_validity(x_cf, target_class)
            l_density = self._loss_density(x_cf, target_class)

            # Total loss
            loss = (l_prox + l_sparse
                    + self.lambda_valid * l_valid
                    + self.lambda_density * l_density)

            if self.use_dtw and neighbours is not None:
                l_dtw = self._loss_dtw_plausibility(x_cf, neighbours)
                loss = loss + self.lambda_dtw * l_dtw
            else:
                l_dtw = torch.tensor(0.0)

            loss.backward()
            optimizer.step()

            loss_history["total"].append(loss.item())
            loss_history["proximity"].append(l_prox.item())
            loss_history["sparsity"].append(l_sparse.item())
            loss_history["validity"].append(l_valid.item())
            loss_history["density"].append(l_density.item())
            loss_history["dtw"].append(l_dtw.item() if isinstance(l_dtw, torch.Tensor) else l_dtw)

            if verbose and (it + 1) % 50 == 0:
                with torch.no_grad():
                    cf_pred = self.classifier.predict(x_cf.unsqueeze(0)).item()
                    cf_prob = self.classifier.predict_proba(
                        x_cf.unsqueeze(0)
                    )[0, target_class].item()
                print(
                    f"  Iter {it+1:4d} | Loss: {loss.item():.4f} | "
                    f"Prox: {l_prox.item():.4f} | Valid: {l_valid.item():.4f} | "
                    f"Density: {l_density.item():.4f} | DTW: {l_dtw.item():.4f} | "
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
        The density model (VAE + GMM) is trained once on X_train, then
        reused for all CFE generations.

        Args:
            X: (n, d, T) — test samples.
            y: (n,) — true labels.
            X_train: (n_train, d, T) — training data (numpy).
            y_train: (n_train,) — training labels (numpy).
            verbose: Print progress.

        Returns:
            List of result dicts (one per sample).
        """
        # Train density model if not already trained
        if self.vae is None or self.gmm is None:
            self._train_density_model(X_train, y_train, verbose=verbose)

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
                print(f"  [GDFO] [{i+1}/{n}] Generating CFE: "
                      f"class {yi} -> {target_class}")

            result = self.generate(xi, target_class, target_samps, verbose=False)
            results.append(result)

        return results
