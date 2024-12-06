import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
from tqdm import tqdm


def z_score_norm(df_data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Function to transform cell values into z-scores for the respective feature.

    Args:
        df_data (pd.DataFrame): input DataFrame to be normalized.

    Returns:
        tuple[pd.DataFrame, StandardScaler]: Normalized data, scaler used for standardizing
        (might be used for testing)
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_data)
    # Create a new DataFrame with the normalized data
    df_normalized = pd.DataFrame(normalized_data, columns=df_data.columns)
    return df_normalized, scaler


def cluster_kmpp(df_data: pd.DataFrame, k: int, random_state=42, n_jobs=-2) -> np.ndarray:
    """Function to cluster data in a pandas DataFrame. Please drop any non-numerical
    columns before. If the data should be normalized, please also do this before.

    Args:
        df_data (pd.DataFrame): numerical data being clustered.
        k (int): Number of clusters.
        random_state (int, optional): Random seed for clustering. Defaults to 42.

    Raises:
        ValueError: If non numerical features are in the DataFrame.
        ValueError: If any NaN values are in the DataFrame.

    Returns:
        np.ndarray: _description_
    """
    non_numerical_features = df_data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
    if non_numerical_features:
        raise ValueError(
            "There appear to be columns that are not numerical (cannot be clustered):"\
            f" {non_numerical_features}"
        )

    if df_data.isnull().values.any():
        raise ValueError("Error, The DataFrame still contained NaN values.")
    with parallel_backend('threading', n_jobs=n_jobs):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state)
        kmeans.fit(df_data.values)

    cluster_labels = kmeans.labels_
    return kmeans, cluster_labels


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def evaluate_clustering(df_data, cluster_sizes=None):
    """
    Evaluates clustering performance over multiple cluster sizes.
    
    Parameters:
    - data (ndarray or DataFrame): The data to cluster.
    - cluster_sizes (list): List of cluster sizes to evaluate.
    
    Returns:
    - results (DataFrame): A DataFrame containing BIC, AIC, and WSS for each cluster size.
    - fig (Figure): A matplotlib Figure object for the generated plots.
    """
    if cluster_sizes is None:
        cluster_sizes = [10] + list(range(50, 1000, 50))  # Default cluster sizes

    from sklearn.metrics import pairwise_distances_argmin_min
    import pandas as pd

    # Initialize lists to store results
    bic_values = []
    aic_values = []
    wss_values = []

    # Evaluate clustering for each cluster size
    for k in tqdm(cluster_sizes):
        # Fit Gaussian Mixture Model for BIC and AIC
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(df_data)
        bic_values.append(gmm.bic(df_data))
        aic_values.append(gmm.aic(df_data))

        # Fit KMeans for Total Within-Cluster Sum of Squares
        kmeans, cluster = cluster_kmpp(df_data, k=k, random_state=42)
        _, distances = pairwise_distances_argmin_min(kmeans.cluster_centers_, df_data)
        wss = sum(distances ** 2)
        wss_values.append(wss)

    # Compile results into a DataFrame
    results = pd.DataFrame({
        'Cluster_Size': cluster_sizes,
        'BIC': bic_values,
        'AIC': aic_values,
        'WSS': wss_values
    })

    # Plot the results
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax[0].plot(cluster_sizes, results['BIC'], marker='o', label='BIC')
    ax[0].set_title('Bayesian Information Criterion (BIC)')
    ax[0].set_xlabel('Cluster Size')
    ax[0].set_ylabel('BIC')
    ax[0].grid(True)

    ax[1].plot(cluster_sizes, results['AIC'], marker='o', label='AIC', color='orange')
    ax[1].set_title('Akaike Information Criterion (AIC)')
    ax[1].set_xlabel('Cluster Size')
    ax[1].set_ylabel('AIC')
    ax[1].grid(True)

    ax[2].plot(cluster_sizes, results['WSS'], marker='o', label='WSS', color='green')
    ax[2].set_title('Total Within-Cluster Sum of Squares (WSS)')
    ax[2].set_xlabel('Cluster Size')
    ax[2].set_ylabel('WSS')
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

    return results, fig
