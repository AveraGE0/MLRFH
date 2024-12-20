from src.dkm import DKNLoss, DataLoader, Autoencoder, training
from torch import optim
from src.dataset import AmsICUSepticShock
import numpy as np
import pandas as pd


def test_dkm():
    np.random.seed(42)
    num_rows = 20
    num_features = 40
    # Create random values for the dataset
    data = np.random.rand(num_rows, num_features)
    # Create column names (e.g., feature_1, feature_2, ..., feature_40)
    columns = [f"feature_{i+1}" for i in range(num_features)]
    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)


    batch_size = 16
    epochs=2
    criterion = DKNLoss()
    input_dim = len(df.columns)
    latent_dim = 20
    autoencoder = Autoencoder(input_dim, latent_dim=latent_dim, layer_sizes=[32, 16, 10], k=400)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    icu_dataset = AmsICUSepticShock(df[columns])
    data_loader = DataLoader(icu_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # Tracked data/metrics

    metrics = training(
        autoencoder,
        optimizer,
        criterion,
        data_loader,
        epochs=epochs,
        model_path='./models/test_model.pth',
        verbose=True
    )
    assert autoencoder.cluster_centers[0][0] != 0
