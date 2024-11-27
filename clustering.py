from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_covtype
from itertools import combinations
from sklearn.preprocessing import StandardScaler

# Task 1
np.random.seed(42)
data = fetch_covtype()
X_orig = data.data
y = data.target

subset_indices = np.random.choice(X_orig.shape[0], size=10000, replace=False)
X_subset = X_orig[subset_indices]
y_subset = y[subset_indices]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)


# Task 2
def kmeans_clustering(X, n_clusters=7):
    kmeans = KMeans(n_clusters=n_clusters,
                    init='random',
                    n_init= 'auto',
                    max_iter=100,
                    tol= 1e-4,
                    random_state=42).fit(X)
    cluster_assignments = kmeans.predict(X)
    return cluster_assignments


# Task 3
def gmm_clustering(X, K=7):
    gmm = GaussianMixture(n_components=K,
                          max_iter=100,
                          covariance_type="full",
                          init_params='random',
                          tol=1e-4).fit(X)
    cluster_assignments = gmm.predict(X)
    return cluster_assignments


# Task 4
def random_baseline(X, K=7):
    random_assignments = np.random.choice(K, size=X.shape[0], replace=True)
    return random_assignments


# Task 5
def count_errors(labels, y_true):
    errors = 0
    total_pairs = 0
    for i, j in combinations(range(len(y_true)), 2):
        if y_true[i] == y_true[j]:
            total_pairs += 1
            if labels[i] != labels[j]:
                errors += 1
    return errors, total_pairs


def perform_pca_and_normalize(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reduced)

    return X_normalized, pca



def main():
    # Performing PCA
    n_components = 44
    X_pca_normalized, pca_model = perform_pca_and_normalize(X_subset, n_components)

    # K-means
    kmeans_assignments = kmeans_clustering(X_pca_normalized)
    kmeans_errors, total_pairs = count_errors(kmeans_assignments, y_subset)
    print("Total number of pairs with the same class label:", total_pairs)
    print("Number of errors made by K-means on PCA-reduced data:", kmeans_errors)
    print("Error rate (%):", (kmeans_errors / total_pairs) * 100)

   #GMM
    gmm_assignments = gmm_clustering(X_pca_normalized)
    gmm_errors, _ = count_errors(gmm_assignments, y_subset)
    print("Number of errors made by GMM on PCA-reduced data:", gmm_errors)
    print("Error rate (%):", (gmm_errors / total_pairs) * 100)

    #Random
    random_assignments = random_baseline(X_pca_normalized, 7)
    random_errors , _ = count_errors(random_assignments, y_subset)
    print("Number of errors made by Random Baseline on PCA-reduced data:", random_errors)
    print("Error rate (%):", (random_errors / total_pairs) * 100)



if __name__ == "__main__":
    main()
