import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns

def cluster_driver_behavior(input_csv='./output/f1_data_final_clean.csv'):
    # Create output directories
    fig_path = './output/clustering/figures'
    res_path = './output/clustering/results'
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(res_path, exist_ok=True)

    # 1. Load qualifying data
    df = pd.read_csv(input_csv)
    qualifying_data = df[df['SessionType'].str.contains('Q')]
    features = qualifying_data[['ThrottleAvg', 'BrakeAvg', 'SpeedAvg']].dropna()

    # 2. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 3. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")

    # 4. Elbow method
    inertia = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_pca)
        inertia.append(km.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 10), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig(f'{fig_path}/elbow_method.png', dpi=300)
    plt.close()

    # 5. Final clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    qualifying_data = qualifying_data.reset_index(drop=True)
    qualifying_data['Cluster'] = clusters

    # 6. Validation via hierarchical clustering
    Z = linkage(X_pca, method='ward')
    coph_corr, _ = cophenet(Z, pdist(X_pca))
    print(f"Cophenetic Correlation: {coph_corr:.2f}")

    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True)
    plt.title(f'Hierarchical Clustering Dendrogram\nCophenetic Correlation: {coph_corr:.2f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/dendrogram_clusters.png', dpi=300)
    plt.close()

    # 7. PCA biplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=50)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Cluster Centers')
    for i, feature in enumerate(['Throttle', 'Brake', 'Speed']):
        plt.arrow(0, 0, pca.components_[0, i]*2, pca.components_[1, i]*2, color='r', width=0.01)
        plt.text(pca.components_[0, i]*2.2, pca.components_[1, i]*2.2, feature, color='r')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA Biplot with Behavioral Clusters and Feature Arrows")
    plt.legend(title='Cluster')
    plt.savefig(f'{fig_path}/pca_biplot_clusters.png', dpi=300)
    plt.close()

    # 8. Telemetry metrics per cluster
    cluster_summary = qualifying_data.groupby('Cluster')[['ThrottleAvg', 'BrakeAvg', 'SpeedAvg']].mean()
    print("\nCluster Averages (Telemetry):\n", cluster_summary)

    # 9. Save outputs
    qualifying_data.to_csv(f'{res_path}/f1_data_with_clusters.csv', index=False)
    cluster_summary.to_csv(f'{res_path}/cluster_averages.csv')

    print(f"\nSaved clustered qualifying data and summaries to {res_path}")

if __name__ == "__main__":
    print("Running driver behavior clustering...")
    cluster_driver_behavior()
