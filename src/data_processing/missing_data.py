import matplotlib.pyplot as plt


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
