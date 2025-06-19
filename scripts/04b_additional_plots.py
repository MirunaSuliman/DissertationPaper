import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_cluster_evolution(input_csv='./output/clustering/results/f1_data_with_clusters.csv'):
    # Create output folder if it doesn't exist
    fig_path = './output/clustering/figures'
    os.makedirs(fig_path, exist_ok=True)

    # Load clustered qualifying data
    df = pd.read_csv(input_csv)

    # Plot cluster distribution by year
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Year', hue='Cluster', palette='Set2')
    plt.title('Distribution of Driving Style Clusters by Season')
    plt.xlabel('Season')
    plt.ylabel('Lap Count')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/cluster_distribution_by_season.png', dpi=300)
    plt.close()

    print(f"Saved longitudinal cluster plots to {fig_path}")

if __name__ == "__main__":
    plot_cluster_evolution()
