import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from data_load import load_data


def correlation_matrix(data_full):
    """
    Plot the correlation matrix of the data
    :param data_full: the full dataset to plot
    :return: None
    """
    # Drop categorical columns
    data_no_categorical = data_full.drop(columns=['Area', 'VehBrand', 'VehGas', 'Region'])
    # Remove outliers from viz, most values are zero, so we remove the zero and top 0.001% of values
    # data_no_categorical = data_no_categorical[data_no_categorical['ClaimAmount']
    #                                           < data_no_categorical['ClaimAmount'].quantile(0.99999)]
    # data_no_categorical = data_no_categorical[data_no_categorical['ClaimAmount'] > 0]

    data_no_categorical['ClaimAmount'] = np.log1p(data_no_categorical['ClaimAmount'])

    corr = data_no_categorical.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

    sns.pairplot(data_no_categorical)
    plt.show()


def plot_claim_amount_relations(data_full):
    """
    Plot the relationship between the features and the target variable (ClaimAmount/Exposure)
    and the distribution of the input features
    :param data_full:
    :return:
    """
    # Remove highest 0.001% values to make the plot more readable
    # data_plot = data_full[data_full['ClaimAmount'] < data_full['ClaimAmount'].quantile(0.99999)]

    data_plot = data_full
    # Remove outliers from viz, most values are zero, so we remove the zero and top 0.001% of values
    # data_plot = data_plot[data_plot['ClaimAmount'] < data_plot['ClaimAmount'].quantile(0.99999)]
    data_plot = data_plot[data_plot['ClaimAmount'] > 0]

    data_plot['ClaimAmount'] = np.log1p(data_plot['ClaimAmount'])

    # Plot the claim amount distribution
    sns.histplot(data_plot['ClaimAmount'], kde=True)
    plt.xlabel('LogClaimAmount')
    plt.ylabel('Density')
    plt.title(f'Total Claim Amount: {data_full["ClaimAmount"].sum():.2f}')
    plt.show()

    for column in data_plot.columns:
        if column != 'ClaimAmount' and column != 'IDpol':
            if column not in ['Area', 'VehBrand', 'VehGas', 'Region']:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                sns.scatterplot(x=data_plot[column], y=data_plot['ClaimAmount'], ax=axes[0])
                axes[0].set_xlabel(column)
                axes[0].set_ylabel('LogClaimAmount')

                sns.histplot(data_plot[column], kde=True, ax=axes[1])
                axes[1].set_xlabel(column)
                axes[1].set_ylabel('Density')

                # Add mean and median lines for n equally spaced points between min and max to see trends
                n_points = 6
                min_val = data_plot[column].min()
                max_val = data_plot[column].max()
                segment_edges = np.linspace(min_val, max_val, n_points + 1)  # Define edges for segments
                segment_centers = (segment_edges[:-1] + segment_edges[1:]) / 2  # Midpoints of segments
                mean_vals = []
                median_vals = []

                for i in range(len(segment_edges) - 1):
                    # Filter data within the current segment
                    segment_data = data_plot[(data_plot[column] >= segment_edges[i]) &
                                             (data_plot[column] < segment_edges[i + 1])]
                    mean_vals.append(segment_data['ClaimAmount'].mean())
                    median_vals.append(segment_data['ClaimAmount'].median())

                axes[0].plot(segment_centers, mean_vals, 'r--', label='Mean')
                axes[0].plot(segment_centers, median_vals, 'g-', label='Median')
                axes[0].legend()

                plt.tight_layout()
                plt.show()
            else:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                sns.boxplot(x=column, y='ClaimAmount', data=data_plot, ax=axes[0])
                axes[0].set_xlabel(column)
                axes[0].set_ylabel('ClaimAmount')

                sns.countplot(x=column, data=data_plot, ax=axes[1])
                axes[1].set_xlabel(column)
                axes[1].set_ylabel('Count')

                plt.tight_layout()
                plt.show()


def summarize_data(data_full):
    """
    Summarize the data
    :param data_full: the full dataset to summarize
    :return: None
    """
    print(data_full.describe())
    print(data_full.info())
    print(data_full.isnull().sum())


if __name__ == "__main__":
    data = load_data()
    summarize_data(data)
    plot_claim_amount_relations(data)
    correlation_matrix(data)
