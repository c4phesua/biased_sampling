"""Conditional Variational Autoencoder, per Ichter et al. 2018"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class CVAE(nn.Module):
    """Conditional Variational Encoder (CVAE)"""

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        latent_dim: int = 16,
        hidden_dims: Sequence[int] = (256, 256),
        beta: float = 1e-3,
        device: Optional[str] = None,
    ):
        super(CVAE, self).__init__()

        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if condition_dim < 0:
            raise ValueError("condition_dim cannot be negative.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if beta < 0:
            raise ValueError("beta cannot be negative.")

        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.beta = beta

        # Encoder
        self.encoder = self._build_mlp(
            state_dim + condition_dim,
            self.hidden_dims,
            2 * latent_dim
        )

        # Decoder
        self.decoder = self._build_mlp(
            latent_dim + condition_dim,
            reversed(self.hidden_dims),
            state_dim,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.ReLU(),
            ])
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def _prepare_condition(self, y: torch.Tensor, batch_size: int):
        """Convert conditions into a 2D tensor"""
        if y is None:
            if self.condition_dim != 0:
                raise ValueError("Condition vector is required when condition_dim is positive")
            return torch.empty(batch_size, 0, device=self.device)

        if y.ndim == 1:
            y = y.unsqueeze(0)

        if y.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dimension {self.condition_dim}, "
                f"got {y.shape[-1]}."
            )

        if y.shape[0] == 1 and batch_size > 1:
            y = y.expand(batch_size, -1)

        if y.shape[0] != batch_size:
            raise ValueError("Condition batch size does not match input batch size.")

        return y

    def encode(self, x, y):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        y = self._prepare_condition(y, x.shape[0])
        encoder_output = self.encoder(torch.cat([x, y], dim=-1))
        mu, logvar = torch.chunk(encoder_output, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor):
        if z.ndim == 1:
            z = z.unsqueeze(0)
        y = self._prepare_condition(y, z.shape[0])
        return self.decoder(torch.cat([z, y], dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def _loss_fn(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: Optional[float] = None
    ):
        if beta is None:
            beta = self.beta
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction="sum") / x.shape[0]
        kl_divergence = -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        ) / x.shape[0]
        total_loss = reconstruction_loss + beta * kl_divergence
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_divergence
        }

    def fit(
        self,
        states: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        beta: Optional[float] = None,
        shuffle: bool = True,
        verbose: bool = True,
    ):
        """
        Train the CVAE.

        Parameters
        ----------
        states:
            Array of shape (N, state_dim).
        conditions:
            Array of shape (N, condition_dim).
        """
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")

        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"Expected states with shape (N, {self.state_dim})")

        if conditions is None:
            if self.condition_dim != 0:
                raise ValueError("Training conditions are required")
            conditions = np.empty((len(states), 0), dtype=np.float32)
        else:
            conditions = np.asarray(conditions, dtype=np.float32)
        if conditions.shape != (len(states), self.condition_dim):
            raise ValueError(
                f"Expected conditions with shape "
                f"({len(states)}, {self.condition_dim}), "
                f"got {conditions.shape}."
            )

        dataset = TensorDataset(torch.from_numpy(states), torch.from_numpy(conditions))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history = {
            "loss": [],
            "reconstruction_loss": [],
            "kl_divergence": [],
        }

        self.train()
        for epoch in range(epochs):
            epoch_losses = {
                "loss": 0.0,
                "reconstruction_loss": 0.0,
                "kl_divergence": 0.0,
            }
            for batch_states, batch_conds in loader:
                batch_states = batch_states.to(self.device)
                batch_conds = batch_conds.to(self.device)
                optimizer.zero_grad()
                reconstruction, mu, logvar = self(batch_states, batch_conds)
                losses = self._loss_fn(reconstruction, batch_states, mu, logvar, beta)
                losses["loss"].backward()
                optimizer.step()
                for key in epoch_losses.keys():
                    epoch_losses[key] += losses[key].item()

            num_batches = len(loader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
                history[key].append(epoch_losses[key])

            if verbose and (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or epoch == epochs - 1
            ):
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Loss: {epoch_losses['loss']:.6f} | "
                    f"Recon: {epoch_losses['reconstruction_loss']:.6f} | "
                    f"KL: {epoch_losses['kl_divergence']:.6f}"
                )

        self.eval()
        return history        

    @torch.no_grad()
    def generate(self, y, n_samples=1, generator=None):
        """
        Generate state from conditional prior
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        y = self._prepare_condition(y, n_samples)
        z = torch.randn(n_samples, self.latent_dim, device=self.device, generator=generator)
        self.eval()
        samples = self.decode(z, y)
        return samples.cpu().numpy()

    def save(self, path):
        torch.save(
            {
                "state_dim": self.state_dim,
                "condition_dim": self.condition_dim,
                "latent_dim": self.latent_dim,
                "hidden_dims": self.hidden_dims,
                "beta": self.beta,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location=device or "cpu")
        model = cls(
            state_dim=checkpoint["state_dim"],
            condition_dim=checkpoint["condition_dim"],
            latent_dim=checkpoint["latent_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            beta=checkpoint["beta"],
            device=device,
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model