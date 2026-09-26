"""
Glacier: Guided Locally Constrained Counterfactual Explanations for
Time Series Classification.

Reference: Wang et al. (2024) — Reference [26] in the paper.

Glacier uses autoencoders to perturb time series within the latent space,
optimising validity and proximity while enforcing plausibility through
reconstruction constraints.

This implements the "uniform" variant which the original authors showed
offers the best results. Due to method limitations, Glacier is only
evaluated on univariate time series datasets.

Key idea:
  1. Train an autoencoder on training data
  2. Encode the original time series → latent space
  3. Optimise in latent space: validity + proximity + reconstruction
  4. Decode back to input space

The uniform variant applies equal weighting across all time steps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from soft_dtw_cfe.config import DEVICE


class GlacierAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for time series.
    
    Architecture:
      Encoder: Conv1d(1→16) → Conv1d(16→8) → Conv1d(8→4) with ReLU
      Decoder: ConvTranspose1d mirroring the encoder
    """

    def __init__(self, seq_len: int, latent_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class GlacierCFE:
    """
    Glacier counterfactual generation (uniform variant).
    
    The method:
      1. Trains an autoencoder on the training data
      2. For each query, optimises in latent space to find a
         counterfactual that is valid, proximal, and reconstructable
    
    Args:
        classifier: Trained classifier (frozen).
        seq_len: Length of the time series.
        ae_epochs: Autoencoder training epochs.
        cfe_iterations: Optimisation steps for CFE generation.
        lr: Learning rate for CFE optimisation.
        lambda_recon: Weight for reconstruction loss.
    """

    def __init__(
        self,
        classifier: nn.Module,
        seq_len: int,
        ae_epochs: int = 100,
        cfe_iterations: int = 500,
        lr: float = 0.01,
        lambda_recon: float = 1.0,
        lambda_valid: float = 5.0,
    ):
        self.classifier = classifier
        self.seq_len = seq_len
        self.ae_epochs = ae_epochs
        self.cfe_iterations = cfe_iterations
        self.lr = lr
        self.lambda_recon = lambda_recon
        self.lambda_valid = lambda_valid
        self.autoencoder = None

        # Freeze classifier
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad_(False)

    def train_autoencoder(self, X_train: np.ndarray, verbose: bool = True):
        """
        Train the autoencoder on training data.
        
        Args:
            X_train: (n, 1, T) — univariate training time series.
        """
        self.autoencoder = GlacierAutoencoder(self.seq_len).to(DEVICE)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

        for epoch in range(self.ae_epochs):
            self.autoencoder.train()
            optimizer.zero_grad()
            recon = self.autoencoder(X_tensor)
            loss = criterion(recon, X_tensor)
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    AE Epoch {epoch+1}/{self.ae_epochs} | Recon Loss: {loss.item():.6f}")

        self.autoencoder.eval()

    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        verbose: bool = False,
    ) -> dict:
        """
        Generate a Glacier counterfactual for a univariate series.
        
        Args:
            x: (1, T) — original univariate time series.
            target_class: Integer label of desired class.
        
        Returns:
            dict with 'counterfactual', 'original', etc.
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not trained. Call train_autoencoder() first.")

        x = x.to(DEVICE)

        with torch.no_grad():
            orig_pred = self.classifier.predict(x.unsqueeze(0)).item()

        # Encode to latent space
        with torch.no_grad():
            z = self.autoencoder.encode(x.unsqueeze(0))  # (1, C_latent, T)

        # Optimise in latent space
        z_cf = z.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z_cf], lr=self.lr)

        for it in range(self.cfe_iterations):
            optimizer.zero_grad()

            # Decode
            x_cf = self.autoencoder.decode(z_cf).squeeze(0)  # (1, T)

            # Validity: hinge loss on target class probability
            probs = self.classifier.predict_proba(x_cf.unsqueeze(0))
            l_valid = torch.relu(0.5 - probs[0, target_class])

            # Proximity (uniform weighting)
            l_prox = (x_cf - x).pow(2).mean()

            # Reconstruction consistency
            z_recon = self.autoencoder.encode(x_cf.unsqueeze(0))
            l_recon = (z_recon - z_cf).pow(2).mean()

            loss = l_prox + self.lambda_valid * l_valid + self.lambda_recon * l_recon
            loss.backward()
            optimizer.step()

        # Final counterfactual
        with torch.no_grad():
            x_cf_final = self.autoencoder.decode(z_cf).squeeze(0).detach()  # (1, T)
            cf_pred = self.classifier.predict(x_cf_final.unsqueeze(0)).item()

        return {
            "counterfactual": x_cf_final.cpu(),
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
        # Train autoencoder if not already trained
        if self.autoencoder is None:
            print("  [Glacier] Training autoencoder...")
            self.train_autoencoder(X_train, verbose=verbose)

        results = []
        n = X.shape[0]

        for i in range(n):
            xi = X[i].to(DEVICE)
            yi = y[i].item() if isinstance(y[i], torch.Tensor) else y[i]

            # Choose target class
            with torch.no_grad():
                probs = self.classifier.predict_proba(xi.unsqueeze(0))[0]
                probs_copy = probs.clone()
                probs_copy[yi] = -1.0
                target_class = probs_copy.argmax().item()

            if verbose and (i + 1) % 20 == 0:
                print(f"  [Glacier] [{i+1}/{n}] class {yi} -> {target_class}")

            result = self.generate(xi, target_class, verbose=False)
            results.append(result)

        return results
