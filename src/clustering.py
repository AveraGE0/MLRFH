import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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


def cluster_kmpp(df_data: pd.DataFrame, k: int, random_state=42) -> np.ndarray:
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

    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state)
    kmeans.fit(df_data.values)

    cluster_labels = kmeans.labels_
    return cluster_labels
