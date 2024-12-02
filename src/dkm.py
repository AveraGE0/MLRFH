"""Module providing code for the deep k-means algorithm."""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def g(
        f: callable,
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
        torch.Tensor: softmax value for index k.
    """
    numerator = torch.exp(-alpha*f(h_x, R[k]))
    denominator = torch.sum(torch.exp(-alpha*f(h_x, r_k)) for r_k in R)
    return numerator/denominator


# Define a basic autoencoder architecture
class Autoencoder(nn.Module):
    """Default Autoencoder architecture. The model contains a bottleneck (latent)
    which is then reconstructed in the decoder. The encoder/decoder are defined
    with similar but mirrored structure of layers (and layer sizes).
    """
    def __init__(self, input_dim: int, latent_dim: int, layer_sizes: list[int]=[64, 32]):
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

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encoder(x)
        decoded = self.decoder(latents)
        return decoded, latents

    def encode(self, x) -> torch.Tensor:
        return self.encoder(x)


class DKNLoss(nn.Module):
    def __init__(self, k, latent_size, lambda_cl=1.0):
        super().__init__()
        self.lambda_cl = lambda_cl
        self.mse_rec = nn.MSELoss()
        self.mse_clust = nn.MSELoss()
        self.k = k
        # we will save the cluster centers in the training
        # to see how they develop
        self.cluster_centers = torch.rand(size=(k, latent_size))
        self.cluster_assignments = None

    def forward(self, x: torch.Tensor, h_x: torch.Tensor, a_x: torch.Tensor):
        """Method to determine the loss of the DKN.

        Args:
          x (torch.Tensor): input data
          h_x (torch.Tensor): latent of input data
          a_x (torch.Tensor): reconstructed input data
        """
        loss_reconstr = self.mse_rec(x, a_x)

        distances = torch.cdist(h_x, self.cluster_centers)
        self.cluster_assignments = torch.argmin(distances, dim=1)  # Closest cluster center
        r_x = self.cluster_centers[self.cluster_assignments]

        loss_clustering = self.mse_clust(h_x, r_x)

        return loss_reconstr + self.lambda_cl * loss_clustering


def training(
        autoencoder: nn.Module,
        optimizer,
        criterion: nn.Module,
        data_loader: DataLoader,
        epochs: int,
        model_path: Path,
        verbose=True
    ) -> dict:
    if not str(model_path).endswith(".pth"):
        raise ValueError(f"Model path should end with .pth! Currently: {model_path}")

    # Tracked metrics
    train_losses = []
    # TODO: add these metrics
    train_loss_reconstruction = []
    train_loss_clusters = []
    cluster_history = []

    for epoch in range(epochs):
        autoencoder.train()
        train_loss = 0

        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()

            reconstructed, latents = autoencoder(batch)
            loss = criterion(batch, latents, reconstructed)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()

            # Update cluster centers
            with torch.no_grad():
                for i in range(criterion.k):
                    # Select latent points assigned to this cluster
                    assigned_points = latents[criterion.cluster_assignments == i]
                    if len(assigned_points) > 0:  # Avoid empty cluster updates
                        criterion.cluster_centers[i] = torch.mean(assigned_points, dim=0)

        cluster_history.append(criterion.cluster_centers.detach().clone())

        train_loss /= len(data_loader)  # Average training loss for the epoch
        train_losses.append(train_loss)

        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {train_loss}")

    # Save the trained model
    torch.save(autoencoder.state_dict(), model_path)

    return {
        "train_loss": train_loss,
        "train_loss_clusters": train_loss_clusters,
        "train_loss_reconstruction": train_loss_reconstruction,
        "cluster_history": cluster_history
    }
