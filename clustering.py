from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_covtype
from matplotlib.patches import Ellipse
from itertools import combinations
from sklearn.preprocessing import StandardScaler

# Load data
np.random.seed(42)
data = fetch_covtype()
X_orig = data.data
y = data.target

# Sample subset of data
subset_indices = np.random.choice(X_orig.shape[0], size=10000, replace=False)
X_subset = X_orig[subset_indices]
y_subset = y[subset_indices]

# Apply PCA for 2D visualization
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X_subset)

# Apply standard scaling to the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)  # Corrected to apply scaling


def plot_clusters(X, cluster_assignments, centroids, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=20, c=cluster_assignments, cmap='viridis', alpha=0.8)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', c='red', edgecolors='k')
    plt.title(title)
    plt.show()


def kmeans_clustering(X, K=7):
    kmeans = KMeans(K, init='random').fit(X)
    cluster_assignments = kmeans.predict(X)
    # centroids = pca.transform(kmeans.cluster_centers_)
    plot_clusters(X_subset, cluster_assignments, kmeans.cluster_centers_, 'K-means Clustering')
    return cluster_assignments


def gmm_clustering(X, K=7):
    gmm = GaussianMixture(n_components=K, max_iter=1000, warm_start=True, init_params='random', tol=1e-8).fit(X)
    cluster_assignments = gmm.predict(X)
    # gmm_means_reduced = pca.transform(gmm.means_)
    plot_clusters(X_subset, cluster_assignments, gmm.means_, 'GMM Clustering')
    return cluster_assignments, gmm


# def random_baseline_clustering(X, K=7):
#     random_assignments = np.random.choice(K, size=X.shape[0], replace=True)
#     random_centroids = np.random.rand(K, 2) * np.ptp(X, axis=0) + np.min(X, axis=0)
#     plot_clusters(X_subset, random_assignments, random_centroids, 'Random Baseline Clustering')
#     return random_assignments

def random_baseline_clustering(X, K=7):
    # Generate random centroids in the original feature space
    random_centroids = np.random.rand(K, X.shape[1]) * np.ptp(X, axis=0) + np.min(X, axis=0)

    # Assign random cluster labels
    random_assignments = np.random.choice(K, size=X.shape[0], replace=True)

    # Plot only the first two features for visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=20, c=random_assignments, cmap='viridis', alpha=0.8)
    plt.scatter(random_centroids[:, 0], random_centroids[:, 1], s=200, marker='X', c='red', edgecolors='k')
    plt.title('Random Baseline Clustering (First 2 Features)')
    plt.show()

    return random_assignments


def count_errors(labels, y_true):
    errors = 0
    total_pairs = 0
    for i, j in combinations(range(len(y_true)), 2):
        if y_true[i] == y_true[j]:
            total_pairs += 1
            if labels[i] != labels[j]:
                errors += 1
    return errors, total_pairs


# Run clustering algorithms and evaluate
kmeans_assignments = kmeans_clustering(X_scaled)
gmm_assignments, gmm_model = gmm_clustering(X_scaled)
random_assignments = random_baseline_clustering(X_scaled)

# Count errors
kmeans_errors, total_pairs = count_errors(kmeans_assignments, y_subset)
gmm_errors, _ = count_errors(gmm_assignments, y_subset)
random_errors, _ = count_errors(random_assignments, y_subset)

print("Total number of pairs with the same label:", total_pairs)
print("K-means errors:", kmeans_errors, "Error rate:", kmeans_errors / total_pairs * 100)
print("GMM errors:", gmm_errors, "Error rate:", gmm_errors / total_pairs * 100)
print("Random baseline errors:", random_errors, "Error rate:", random_errors / total_pairs * 100)

# Print results
print("Total number of pairs with the same class label:", total_pairs)
print("Number of errors made by K-means:", kmeans_errors)
print("Number of errors made by GMM:", gmm_errors)
print("Number of errors made by random baseline:", random_errors)

# Calculate and print silhouette score for GMM
# Calculate Silhouette Score and Calinski-Harabasz Index
silhouette_kmeans = silhouette_score(X_scaled, kmeans_assignments)
silhouette_gmm = silhouette_score(X_scaled, gmm_assignments)
silhouette_random = silhouette_score(X_subset, random_assignments)

calinski_harabasz_kmeans = calinski_harabasz_score(X_subset, kmeans_assignments)
calinski_harabasz_gmm = calinski_harabasz_score(X_subset, gmm_assignments)
calinski_harabasz_random = calinski_harabasz_score(X_subset, random_assignments)

# Print results
print("Silhouette Score for K-Means:", silhouette_kmeans)
print("Silhouette Score for GMM:", silhouette_gmm)
print("Silhouette Score for Random Baseline:", silhouette_random)

print("Calinski-Harabasz Index for K-Means:", calinski_harabasz_kmeans)
print("Calinski-Harabasz Index for GMM:", calinski_harabasz_gmm)
print("Calinski-Harabasz Index for Random Baseline:", calinski_harabasz_random)