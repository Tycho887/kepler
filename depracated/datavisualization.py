import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path = "datasets/tle11175.csv"):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)


# We want to visualize the dataset to understand the distribution of the features and the target variable.

def visualize_correlation(data: pd.DataFrame):
    """
    Visualize the dataset to understand the distribution of features and target variable.

    Parameters:
        data (pd.DataFrame): Dataset containing features and target variable.
    """

    # Create correlation matrix

    corr = data.corr()

    # Plot heatmap of correlation matrix

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig("images/correlation_matrix.png")
    plt.close()


def plot_target_distribution(data: pd.DataFrame, target_column: str):
    """
    Visualize the distribution of the target variable.

    Parameters:
        data (pd.DataFrame): Dataset containing features and target variable.
        target_column (str): Name of the target variable column.

    Returns:
        None: Displays the plot.
    """
    target_counts = data[target_column].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_counts.index, y=target_counts.values, palette="viridis")
    plt.title("Distribution of Target Variable", fontsize=16)
    plt.xlabel("Target Value", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(ticks=[0, 1], labels=["False", "True"], fontsize=12)
    plt.tight_layout()
    plt.savefig("images/target_distribution.png")
    plt.close()


if __name__ == "__main__":
    # Load the dataset
    data = load_dataset()

    # Visualize the dataset
    visualize_correlation(data)

    # Plot the distribution of the target variable

    plot_target_distribution(data, "close_approach")