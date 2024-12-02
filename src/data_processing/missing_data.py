import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_missing_data(df, imputation_type):
    """
    Plot the percentage of missing data for each feature in the dataframe and return the figure.

    Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        imputation_type (str): A description of the imputation type to include in the plot title.

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
    """
    columns_to_check = df.columns[:]

    # Calculate the percentage of missing data per feature
    missing_percentage = df[columns_to_check].isnull().mean() * 100

    # Display the results
    print("Percentage of Missing Data Per Feature:")
    print(missing_percentage)

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_percentage.sort_values(ascending=False).plot(kind="bar", color="blue", alpha=0.7, ax=ax)
    ax.set_title(f"Percentage of Missing Data Per Feature {imputation_type}")
    ax.set_ylabel("Missing Data (%)")
    ax.set_xlabel("Features")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return fig


def calculate_average_nans(column):
    # Get the positions of non-NaN values as numeric positions
    non_nan_positions = np.where(~column.isna())[0]  # Use numpy to extract positions

    # Calculate gaps between non-NaN values
    if len(non_nan_positions) > 1:
        gaps = np.diff(non_nan_positions) - 1  # Subtract 1 to count gaps between positions
        return np.mean(gaps) if len(gaps) > 0 else 0
    else:
        return 0  # No gaps if fewer than 2 non-NaN values


def drop_na_cols(df_data, threshold=0.30):
    before_filtering = df_data.index.unique().shape[0]  # Get unique index values
    # Define threshold for minimum data availability
    threshold = 0.30

    # Identify numeric columns to evaluate (columns after 'year_of_birth')
    numeric_cols = df_data.columns[:]

    # Calculate the percentage of available data per visit occurrence
    availability = (
        df_data.groupby(level=[0, 1])[numeric_cols]  # Group by index levels
        .apply(lambda group: group.notna().mean().mean())  # Mean across columns, then rows
    )

    # Identify visit occurrences to drop
    to_drop = availability[availability < threshold].index

    # Filter the dataset
    # Assuming 'df_sepsis_wide' has 'person_id' and 'visit_occurrence_id' as columns
    df_filtered = df_data[~df_data.index.isin(to_drop)] # Filter based on result index

    # Count the number of unique visit occurrences after filtering
    after_filtering = df_filtered.index.unique().shape[0]  # Get unique index values
    # Display the percentage of visit occurrences retained
    retained_percentage = (after_filtering / before_filtering) * 100
    print(f"Percentage of visit occurrences retained: {retained_percentage:.2f}%")
    print(f"Number of visit occurrences after filtering: {after_filtering}")
    print(f"Dropped the following columns: {to_drop}")
    return df_filtered

