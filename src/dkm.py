"""Module providing code for the deep k-means algorithm."""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def g(
        k: int,
        h_x: torch.Tensor,
        alpha: float,
        R: torch.Tensor
    ) -> torch.Tensor:
    """The G_{f,k} function from the Deep K-Means paper.
    Calculates a softmax smoothed by alpha.

    Args:
        f (callable): distance function (for cluster center distance).
        k (int): Cluster index (numerator).
        h_x (torch.Tensor): Latent representation.
        alpha (float): Smoothness parameter.
        R (torch.Tensor): (Current) cluster centers.

    Returns:
        torch.Tensor: Softmax value for index k.
    """
    distances = torch.cdist(h_x.unsqueeze(0), R.unsqueeze(0)).squeeze(0)  # Pairwise distances, replaces f
    numerator = torch.exp(-alpha * distances[:, k])
    denominator = torch.sum(torch.exp(-alpha * distances), dim=1) + 1e-8  # Add epsilon for numerical stability
    return numerator / denominator


# Define a basic autoencoder architecture
class Autoencoder(nn.Module):
    """Default Autoencoder architecture. The model contains a bottleneck (latent)
    which is then reconstructed in the decoder. The encoder/decoder are defined
    with similar but mirrored structure of layers (and layer sizes).
    """
    def __init__(self, input_dim: int, latent_dim: int, k: int, layer_sizes: list[int]=[64, 32]):
        super().__init__()
        self.latent_dim = latent_dim

        layer_sizes = [input_dim] + layer_sizes + [latent_dim]
        layers_encoder = []
        layers_decoder = []
        layer_amount = len(layer_sizes)

        for i in range(layer_amount-1):
            layers_encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i != layer_amount-2:
                layers_encoder.append(nn.ReLU())

            layers_decoder.append(nn.Linear(layer_sizes[-(i+1)], layer_sizes[-(i+2)]))
            if i != layer_amount-2:
                layers_decoder.append(nn.ReLU())

        self.encoder = nn.Sequential(
            *layers_encoder
        )
        self.decoder = nn.Sequential(
            *layers_decoder
        )
        self.cluster_centers = nn.Parameter(torch.rand(k, latent_dim))  # Learnable cluster centers

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encoder(x)
        decoded = self.decoder(latents)
        return decoded, latents

    def encode(self, x) -> torch.Tensor:
        return self.encoder(x)


class DKNLoss(nn.Module):
    def __init__(self, alpha=1.0, lambda_cl=1.0):
        super().__init__()
        self.lambda_cl = lambda_cl
        self.mse_rec = nn.MSELoss()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, h_x: torch.Tensor, a_x: torch.Tensor, cluster_centers):
        """Compute the DKN loss.

        Args:
          x (torch.Tensor): Input data.
          h_x (torch.Tensor): Latent representations of input data.
          a_x (torch.Tensor): Reconstructed data.
        """
        # Reconstruction Loss
        loss_reconstr = self.mse_rec(x, a_x)

        # Compute soft assignments using G function
        soft_assignments = torch.stack(
            [g(k, h_x, self.alpha, cluster_centers) for k in range(cluster_centers.size(0))],
            dim=1
        )  # Shape: [batch_size, k]

        # Compute weighted cluster centers
        weighted_centers = torch.matmul(soft_assignments, cluster_centers)  # Shape: [batch_size, latent_size]

        # Clustering Loss
        loss_clustering = self.mse_rec(h_x, weighted_centers)

        return loss_reconstr + self.lambda_cl * loss_clustering



def training(
        autoencoder: nn.Module,
        optimizer,
        criterion: nn.Module,
        data_loader: DataLoader,
        epochs: int,
        model_path: Path,
        alpha_init=1.0,
        alpha_final=100.0,
        verbose=True
    ) -> dict:
    if not str(model_path).endswith(".pth"):
        raise ValueError(f"Model path should end with .pth got {model_path}.")

    # Tracked metrics
    train_losses = []

    alpha = alpha_init
    alpha_step = (alpha_final - alpha_init) / epochs

    for epoch in range(epochs):
        autoencoder.train()
        train_loss = 0

        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            batch = batch.to(next(autoencoder.parameters()).device)
            # Forward pass
            reconstructed, latents = autoencoder(batch)
            criterion.alpha = alpha
            loss = criterion(batch, latents, reconstructed, autoencoder.cluster_centers)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(data_loader)  # Average training loss for the epoch
        train_losses.append(train_loss)

        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {train_loss}")

        alpha += alpha_step

        # Save the trained model
        print("Saving model")
        torch.save(autoencoder.state_dict(), model_path)

    return {
        "train_loss": train_losses
    }
