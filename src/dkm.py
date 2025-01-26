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
    distances = torch.cdist(h_x.unsqueeze(0), R.unsqueeze(0)).squeeze(0)
    min_dist = torch.min(distances, dim=1, keepdim=True)[0]
    numerator = torch.exp(-alpha * (distances - min_dist))
    denominator = torch.sum(numerator, dim=1) + 1e-8
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
        self.mse_clust = nn.MSELoss()
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
        # soft_assignments = torch.stack(
        #     [g(k, h_x, self.alpha, cluster_centers) for k in range(cluster_centers.size(0))],
        #     dim=1
        # )  # Shape: [batch_size, k]

        # # Compute weighted cluster centers
        # weighted_centers = torch.matmul(soft_assignments, cluster_centers)  # Shape: [batch_size, latent_size]

        # # Clustering Loss
        # loss_clustering = self.mse_clust(h_x, weighted_centers)
        
        # hard_assignments = torch.argmax(soft_assignments, dim=1)
        # hard_centers = cluster_centers[hard_assignments]
        # loss_clustering_hard = self.mse_clust(h_x, hard_centers)
        
        # Compute pairwise distances between embeddings and cluster centers
        distances = torch.cdist(h_x.unsqueeze(0), cluster_centers.unsqueeze(0)).squeeze(0)  # Shape: [batch_size, n_clusters]

        # Compute soft assignments using G function (softmax with alpha)
        #min_dist = torch.min(distances, dim=1, keepdim=True)[0]
        numerator = torch.exp(-self.alpha * (distances))
        denominator = torch.sum(numerator, dim=1, keepdim=True) + 1e-8  # Shape: [batch_size, 1]
        soft_assignments = numerator / denominator  # Shape: [batch_size, n_clusters]

        # Step 3: Compute weighted distances (element-wise product of distances and soft assignments)
        weighted_distances = distances * soft_assignments  # Shape: [batch_size, n_clusters]

        # Step 4: Sum weighted distances across clusters for each data point
        sum_weighted_distances = torch.sum(weighted_distances, dim=1)  # Shape: [batch_size]

        # Step 5: Compute clustering loss as the mean of these sums
        loss_clustering = torch.mean(sum_weighted_distances)  # Scalar

        return loss_reconstr + self.lambda_cl * loss_clustering, loss_reconstr, loss_clustering



def training(
        autoencoder: nn.Module,
        optimizer,
        criterion: nn.Module,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        model_path: Path,
        alpha_init=0.05,
        alpha_final=100.0,
        max_epochs: int=100,
        pretrain_epochs=30,
        verbose=True
    ) -> dict:
    if not str(model_path).endswith(".pth"):
        raise ValueError(f"Model path should end with .pth got {model_path}.")

    # Tracked metrics
    train_losses = []
    train_losses_clust = []
    train_losses_rec = []

    val_losses = []
    val_losses_rec = []
    val_losses_clust = []

    # pretraining
    pretrain_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)  # Pretraining optimizer
    pretrain_criterion = torch.nn.MSELoss()  # Reconstruction loss for pretraining

    print("Starting pretraining...")
    for epoch in range(pretrain_epochs):
        autoencoder.train()
        epoch_pretrain_loss = 0

        for batch in tqdm(train_data_loader, desc=f"Pretrain Epoch {epoch + 1}/{pretrain_epochs}"):
            pretrain_optimizer.zero_grad()
            batch = batch.to(next(autoencoder.parameters()).device)
            # Forward pass
            reconstructed, _ = autoencoder(batch)
            # Compute reconstruction loss
            loss = pretrain_criterion(reconstructed, batch)
            loss.backward()
            pretrain_optimizer.step()
            epoch_pretrain_loss += loss.item()

        epoch_pretrain_loss /= len(train_data_loader)

        if verbose:
            print(f"Pretraining Epoch {epoch + 1}, Reconstruction Loss: {epoch_pretrain_loss}")

    print("Pretraining completed. Proceeding to full training...")

    alpha = alpha_init

    early_stopping_patience: int = 5
    delta: float = 0.0        # Minimum change in loss to qualify as improvement
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        autoencoder.train()
        epoch_train_loss = 0
        epoch_train_loss_rec = 0
        epoch_train_loss_clust = 0

        criterion.alpha = alpha

        autoencoder.train()
        for batch in tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{max_epochs}"):
            optimizer.zero_grad()
            batch = batch.to(next(autoencoder.parameters()).device)
            # Forward pass
            reconstructed, latents = autoencoder(batch)
            
            total_train_loss, loss_train_rec, loss_train_clust = criterion(batch, latents, reconstructed, autoencoder.cluster_centers)
            
            total_train_loss.backward()
            optimizer.step()
            epoch_train_loss += total_train_loss.item()
            epoch_train_loss_rec += loss_train_rec.item()
            epoch_train_loss_clust += loss_train_clust.item()

        # Average training loss for the epoch
        epoch_train_loss /= len(train_data_loader)
        epoch_train_loss_rec /= len(train_data_loader)
        epoch_train_loss_clust /= len(train_data_loader)

        train_losses.append(epoch_train_loss)
        train_losses_rec.append(epoch_train_loss_rec)
        train_losses_clust.append(epoch_train_loss_clust)

        # Validation phase
        autoencoder.eval()  # Set to evaluation mode
        epoch_val_loss = 0
        epoch_val_loss_rec = 0
        epoch_val_loss_clust = 0

        with torch.no_grad():  # No gradient computation during validation
            for batch in val_data_loader:
                batch = batch.to(next(autoencoder.parameters()).device)
                reconstructed, latents = autoencoder(batch)
                total_val_loss, loss_val_rec, loss_val_clust = criterion(batch, latents, reconstructed, autoencoder.cluster_centers)
                
                epoch_val_loss += total_val_loss.item()
                epoch_val_loss_rec += loss_val_rec.item()
                epoch_val_loss_clust += loss_val_clust.item()


        epoch_val_loss /= len(val_data_loader)
        epoch_val_loss_rec /= len(val_data_loader)
        epoch_val_loss_clust /= len(val_data_loader)

        val_losses.append(epoch_val_loss)
        val_losses_rec.append(epoch_val_loss_rec)
        val_losses_clust.append(epoch_val_loss_clust)

        if verbose:
            print(f"Validation Loss: {epoch_val_loss}")
            print(f"Epoch {epoch + 1}, Loss: {epoch_train_loss}")

        # Early stopping logic
        if epoch_val_loss < best_loss - delta:
            best_loss = epoch_val_loss
            epochs_without_improvement = 0  # Reset counter
            # Save the best model
            torch.save(autoencoder.state_dict(), model_path)
            if verbose:
                print(f"Model improved, saving best model at epoch {epoch + 1}")
        else:
            epochs_without_improvement += 1
            if verbose:
                print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print("Early stopping triggered.")
                break
        
        n = max(epoch, 2)
        alpha = min(2**(1/np.log(n)**2) * alpha, alpha_final)

        # Save the trained model
        #print("Saving model")
        #torch.save(autoencoder.state_dict(), model_path)

    return {
        "train_loss": train_losses,
        "train_loss_rec": train_losses_rec,
        "train_loss_clust": train_losses_clust,
        "val_loss": val_losses,
        "val_loss_rec": val_losses_rec,
        "val_loss_clust": val_losses_clust
    }
