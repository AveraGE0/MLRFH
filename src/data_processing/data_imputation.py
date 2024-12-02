import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


# Example forward-fill implementation with limit
def forward_fill_limited(group, column, average_nans):
    # Create a copy of the column to avoid modifying the original data directly
    col_data = group[column].copy()
    # Apply forward fill with a limit equal to the average NaNs
    # Ensure limit is at least 1 to avoid ValueError
    limit = int(average_nans)
    if limit < 1:
        limit = 1  # Set a minimum limit of 1
    col_data.ffill(limit=limit, inplace=True)
    return col_data

# Perform limited forward fill per column and visit_occurrence_id
def forward_fill_by_column(df, avg_nans_per_column):
    df_filled = df.copy()  # Create a copy to avoid modifying the original data
    for column in df.iloc[:, :].columns:  # Loop over feature columns (after metadata columns)
        if column in avg_nans_per_column.index:  # Ensure the column has an average NaN value
            average_nans = avg_nans_per_column[column]

            # Special handling for 'Body weight' column
            if column == 'Body weight':
                df_filled[column] = df.groupby("visit_occurrence_id", group_keys=False)[column].ffill()
            else:
                df_filled[column] = (
                    df.groupby("visit_occurrence_id", group_keys=False)
                    .apply(lambda group: forward_fill_limited(group, column, average_nans))
                )
    return df_filled

# Calculate average NaNs per column using the previous logic
def calculate_average_nans(column):
    non_nan_positions = np.where(~column.isna())[0]  # Get numeric positions of non-NaN values
    if len(non_nan_positions) > 1:
        gaps = np.diff(non_nan_positions) - 1
        return np.mean(gaps) if len(gaps) > 0 else 0
    else:
        return 0  # No gaps if fewer than 2 non-NaN values
    

# Define a function to apply KNN imputation for each visit occurrence
def knn_impute_by_column(df):
    """
    Applies KNN imputation to each visit_occurrence_id group using dynamically calculated k.
    """
    df_imputed = df.copy()  # Create a copy of the DataFrame
    feature_columns = df.iloc[:, :].columns  # Columns after metadata (features only)

    for visit_id, group in df.groupby("visit_occurrence_id"):
        # Only process the feature columns for this visit occurrence
        group_features = group[feature_columns]

        # Dynamically calculate k based on the number of rows in the group
        N = len(group_features)
        k = int(np.sqrt(N))
        k = max(2, min(k, N - 1))  # Ensure k is between 2 and N-1

        # Initialize KNNImputer with dynamically calculated k
        knn_imputer = KNNImputer(n_neighbors=k)

        # Apply KNNImputer if there are missing values
        if group_features.isna().sum().sum() > 0:
            # Fit-transform the group features
            imputed_values = knn_imputer.fit_transform(group_features)

            # Get the columns actually used for imputation (might be fewer due to all NaNs)
            imputed_columns = group_features.columns[~group_features.isna().all()]

            # Ensure that imputed values align with the original index and corresponding columns
            df_imputed.loc[group.index, imputed_columns] = imputed_values

    return df_imputed


def drop_features_with_missing_data(df, threshold=65):
    """
    Drops columns (features) with more than the given percentage of missing values.

    Parameters:
    - df: DataFrame to process
    - threshold: Percentage threshold for dropping columns (e.g., 40 for 40%)

    Returns:
    - Processed DataFrame with columns dropped
    - List of columns dropped
    """
    # Calculate the percentage of missing data for each column
    missing_percentage = df.isnull().mean() * 100  # Missing percentage per column

    # Identify columns to drop
    columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()

    # Drop the identified columns
    df = df.drop(columns=columns_to_drop)

    return df, columns_to_drop