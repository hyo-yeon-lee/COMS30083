import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_covtype
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import combinations

# Task 1
def load_and_sample_data():
    dataset_path = "covtype.data" # fetch not working ...
    # data_set = fetch_covtype()
    # print(data_set.data.shape)
    print("Loading dataset from local file...")
    data = pd.read_csv(dataset_path, header=None)
    # data = fetch_covtype()
    print(data.shape)
    X = data.iloc[:, :-1].values  # Feature values
    y = data.iloc[:, -1].values   # True class labels
    np.random.seed(42)
    subset_indices = np.random.choice(X.shape[0], 10000, replace=False)
    print(X[subset_indices].shape)

    return X[subset_indices], y[subset_indices]

# Task 2
def kmeans_clustering(X, n_clusters=7):
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42).fit(X)
    cluster_assignments = kmeans.predict(X)
    centroid_locations = kmeans.cluster_centers_
    return cluster_assignments, centroid_locations

# Task 3
def gmm_clustering(X, n_clusters=7):
    #Error rate about 85%
    # gmm = GaussianMixture(n_components=n_clusters, covariance_type = 'spherical',max_iter=2000, warm_start=True, init_params='random', tol=1e-8).fit(X)
    #Error rate about 70%
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        n_init=10,
        max_iter=1500,
        warm_start=False,
        tol=1e-4,
        init_params='random'
    ).fit(X)

    # gmm = BayesianGaussianMixture(
    #     n_components=n_clusters,
    #     covariance_type='full',
    #     max_iter=1000,
    #     n_init=10,
    #     tol=1e-4,
    #     init_params='random'
    # ).fit(X)
    #

    cluster_assignments = gmm.predict(X)
    centroid_locations = gmm.means_

    return cluster_assignments, centroid_locations

# Task 4
def random_baseline_clustering(n_samples, n_clusters=7):
    np.random.seed(42)
    labels = np.random.randint(0, n_clusters, size=n_samples)
    return labels

# Function to plot K-means and GMM clustering results
def plot_clusters(X, cluster_assignments, centroid_locations, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', s=10, label="Data Points")
    plt.scatter(centroid_locations[:, 0], centroid_locations[:, 1], c='red', s=100, marker='X', label="Centroids")
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

# Task 5
def plot_random_baseline(X, random_labels, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=random_labels, cmap='viridis', s=10, label="Data Points")
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

# Task 5
def count_clustering_errors(true_labels, predicted_labels):
    errors = 0
    for i, j in combinations(range(len(true_labels)), 2):
        if true_labels[i] == true_labels[j]:  # Same class label
            if predicted_labels[i] != predicted_labels[j]:  # Different clusters
                errors += 1
    return errors

def count_same_clusters(true_labels):
    same_labels = 0
    # Check pairs of points with the same class label
    for i, j in combinations(range(len(true_labels)), 2):
        if true_labels[i] == true_labels[j]:  # Same class label
            same_labels += 1
    return same_labels

def main():
    # Load data
    X, y = load_and_sample_data()

    kmeans_assignments, kmeans_centroids = kmeans_clustering(X)
    plot_clusters(X, kmeans_assignments, kmeans_centroids, "K-means Clustering")

    gmm_assignments, gmm_centroids = gmm_clustering(X)
    plot_clusters(X, gmm_assignments, gmm_centroids, "Gaussian Mixture Model Clustering")

    random_labels = random_baseline_clustering(len(X))
    plot_random_baseline(X, random_labels, "Random Baseline Clustering")

    kmeans_errors = count_clustering_errors(y, kmeans_assignments)
    gmm_errors = count_clustering_errors(y, gmm_assignments)
    random_errors = count_clustering_errors(y, random_labels)
    same_labels = count_same_clusters(y)

    print("Error rate in K-means clustering:", kmeans_errors / same_labels * 100 , "%")
    print("Error rate in GMM clustering:", gmm_errors / same_labels * 100 , "%")
    print("Error rate in random baseline clustering:", random_errors / same_labels * 100 , "%")
    # print("Number of same labels in clustering:", same_labels)

if __name__ == "__main__":
    main()
