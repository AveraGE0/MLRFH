"""Module containing code for outlier dropping."""
import pandas as pd
import numpy as np
from tqdm import tqdm


def drop_outliers_iqr_wide(df_wide: pd.DataFrame, outlier_columns: list[str]):
    """
    Remove outliers using IQR for specified columns in a wide-format dataset.

    Parameters:
        df_wide (pd.DataFrame): Input DataFrame with wide-format data (one column per feature).
        outlier_columns (list): List of column names where outliers should be removed.

    Returns:
        pd.DataFrame: DataFrame with outliers removed for specified columns.
    """
    for column in tqdm(outlier_columns, desc=f"Processing {len(outlier_columns)} columns"):
        if column not in df_wide.columns:
            print(f"Warning: Column '{column}' not found in the DataFrame. Skipping.")
            continue
        
        # Drop NaN values and calculate IQR
        feature_values = df_wide[column].dropna().astype(float)
        Q1 = np.percentile(feature_values, 25)  # 25th percentile
        Q3 = np.percentile(feature_values, 75)  # 75th percentile
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Drop outliers
        df_wide.loc[
            (df_wide[column] >= lower_bound) &\
            (df_wide[column] <= upper_bound),
            column
        ] = np.nan

    return df_wide


def drop_outliers_iqr_long(df_long: pd.DataFrame, outlier_columns: list[str]) -> pd.DataFrame:
    """Removed outliers from all features, that are outside of the
    1.5 * IQR range.
    Args:
        df_long (pd.DataFrame): DataFrame with the data in the long format.
        outlier_columns (list[str]): List with columns in which the outliers should be removed

    Returns:
        pd.DataFrame: The DataFrame without the outlier values.
    """
    for feature in tqdm(outlier_columns, desc=f"processing f{len(outlier_columns)} features"):
        # Select data for the current concept
        feature_values = np.array(df_long[df_long['feature_name'] == feature]['value_as_number']\
            .dropna()\
            .astype(float)\
            .tolist()
        )
        Q1 = np.percentile(feature_values, 25)  # 25th percentile
        Q3 = np.percentile(feature_values, 75)  # 75th percentile
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter data to remove outliers
        df_long = df_long[
            (df_long['feature_name'] != feature) |
            (
                (df_long["value_as_number"] >= lower_bound) &
                (df_long["value_as_number"] <= upper_bound)
            )
        ]
    return df_long
