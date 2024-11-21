from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_covtype
from itertools import combinations
from sklearn.preprocessing import StandardScaler

# Load data
np.random.seed(42)
data = fetch_covtype()
X_orig = data.data
y = data.target

subset_indices = np.random.choice(X_orig.shape[0], size=10000, replace=False)
X_subset = X_orig[subset_indices]
y_subset = y[subset_indices]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)



def plot_clusters(X, cluster_assignments, centroids, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=20, c=cluster_assignments, cmap='viridis', alpha=0.8)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', c='red', edgecolors='k')
    plt.title(title)
    plt.show()


def kmeans_clustering(X, n_clusters=7):
    kmeans = KMeans(n_clusters=n_clusters,
                    init='random',
                    n_init= 'auto',
                    max_iter=100,
                    random_state=42).fit(X)
    cluster_assignments = kmeans.predict(X)
    centroid_locations = kmeans.cluster_centers_
    return cluster_assignments


def gmm_clustering(X, K=7):
    gmm = GaussianMixture(n_components=K,
                          max_iter=300,
                          covariance_type="full",
                          warm_start=True,
                          init_params='kmeans',
                          tol=1e-8).fit(X)
    cluster_assignments = gmm.predict(X)
    return cluster_assignments



def random_baseline_clustering(X, K=7):
    # Generate random centroids in the original feature space
    random_centroids = np.random.rand(K, X.shape[1]) * np.ptp(X, axis=0) + np.min(X, axis=0)

    random_assignments = np.random.choice(K, size=X.shape[0], replace=True)
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


def perform_pca_and_normalize(X, n_components):
    # Step 1: Perform PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    # Step 2: Normalize the PCA-reduced data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reduced)

    return X_normalized, pca


def evaluate_pca_variance(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, n_components + 1), explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of PCA Components')
    plt.grid(True)
    plt.show()
    return explained_variance


def explained_variance_analysis(X, target_components=44):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=X.shape[1])
    pca.fit(X_scaled)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, X.shape[1] + 1), cumulative_explained_variance, marker='o', label='Cumulative Variance')
    plt.axvline(x=target_components, color='r', linestyle='--', label=f'{target_components} Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Principal Components')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print the variance retained with the target number of components
    print(f"Variance retained with {target_components} components: {cumulative_explained_variance[target_components - 1]:.2f}")

def evaluate_dbi_hyperparameters(X, hyperparameter, values, **kwargs):
    dbi_scores = []
    for value in values:
        kmeans = KMeans(n_clusters=7, random_state=42, **{hyperparameter: value}, **kwargs)
        cluster_assignments = kmeans.fit_predict(X)
        dbi = davies_bouldin_score(X, cluster_assignments)
        dbi_scores.append(dbi)
    return dbi_scores

# Plotting function
def plot_dbi(hyperparameter, values, dbi_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(values, dbi_scores, marker='o')
    plt.xlabel(hyperparameter)
    plt.ylabel('Davies-Bouldin Index')
    plt.title(f'DBI vs {hyperparameter}')
    plt.grid(True)
    plt.show()


def evaluate_gmm_dbi_hyperparameters(X, hyperparameter, values, **kwargs):
    dbi_scores = []
    for value in values:
        gmm = GaussianMixture(n_components=7, random_state=42, **{hyperparameter: value}, **kwargs)
        cluster_assignments = gmm.fit_predict(X)
        dbi = davies_bouldin_score(X, cluster_assignments)
        dbi_scores.append(dbi)
    return dbi_scores

def plot_gmm_dbi(hyperparameter, values, dbi_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(values, dbi_scores, marker='o')
    plt.xlabel(hyperparameter)
    plt.ylabel('Davies-Bouldin Index')
    plt.title(f'GMM: DBI vs {hyperparameter}')
    plt.grid(True)
    plt.show()


def main():
    print("Evaluating explained variance for PCA...")
    explained_variance_analysis(X_orig, target_components=44)
    evaluate_pca_variance(X_subset, n_components=44)

    # Step 2: Perform PCA and normalization
    n_components = 44
    X_pca_normalized, pca_model = perform_pca_and_normalize(X_subset, n_components)

    # covariance_values = ['full', 'tied', 'diag', 'spherical']
    # dbi_covariance = evaluate_gmm_dbi_hyperparameters(X_pca_normalized, 'covariance_type', covariance_values)
    # plot_gmm_dbi('covariance_type', covariance_values, dbi_covariance)
    #
    # # Evaluate DBI for `init_params`
    # init_values = ['kmeans', 'random']
    # dbi_init_params = evaluate_gmm_dbi_hyperparameters(X_pca_normalized, 'init_params', init_values)
    # plot_gmm_dbi('init_params', init_values, dbi_init_params)
    #
    # # Evaluate DBI for `max_iter`
    # max_iter_values = [100, 200, 300, 400, 500]
    # dbi_max_iter = evaluate_gmm_dbi_hyperparameters(X_pca_normalized, 'max_iter', max_iter_values)
    # plot_gmm_dbi('max_iter', max_iter_values, dbi_max_iter)

    # n_init_values = [1, 5, 10, 20, 'auto']
    # dbi_n_init = evaluate_dbi_hyperparameters(X_pca_normalized, 'n_init', n_init_values)
    # plot_dbi('n_init', n_init_values, dbi_n_init)
    #
    # # 2. Evaluate DBI for `init`
    # init_values = ['k-means++', 'random']
    # dbi_init = evaluate_dbi_hyperparameters(X_pca_normalized, 'init', init_values)
    # plot_dbi('init', init_values, dbi_init)
    #
    # # 3. Evaluate DBI for `tol`
    # tol_values = [1e-4, 1e-3, 1e-2]
    # dbi_tol = evaluate_dbi_hyperparameters(X_pca_normalized, 'tol', tol_values)
    # plot_dbi('tol', tol_values, dbi_tol)
    #
    # max_iter = [100, 200, 300, 400, 500]
    # dbi_tol = evaluate_dbi_hyperparameters(X_pca_normalized, 'max_iter', max_iter)
    # plot_dbi('max_iter', max_iter, dbi_tol)
    # # print(f"Shape after PCA: {X_pca_normalized.shape}")
    # # print(f"Explained variance ratio (first few components): {pca_model.explained_variance_ratio_[:5]}")
    # #
    # # # Step 3: Run GMM on the PCA-reduced data
    # print("Running GMM on PCA-reduced data...")
    gmm_assignments = gmm_clustering(X_pca_normalized)
    # #
    # # # Step 4: Evaluate clustering performance for GMM
    # print("Evaluating GMM clustering performance...")
    gmm_errors, total_pairs = count_errors(gmm_assignments, y_subset)
    print("Total number of pairs with the same class label:", total_pairs)
    print("Number of errors made by GMM on PCA-reduced data:", gmm_errors)
    print("Error rate (%):", (gmm_errors / total_pairs) * 100)
    # #
    # # # Step 5: Run K-means on the PCA-reduced data
    # print("Running K-meansc on PCA-reduced data...")
    # kmeans_assignments = kmeans_clustering(X_pca_normalized)
    # #
    # # # Step 6: Evaluate clustering performance for K-means
    # print("Evaluating K-means clustering performance...")
    # kmeans_errors, total_pairs = count_errors(kmeans_assignments, y_subset)
    # print("Number of errors made by K-means on PCA-reduced data:", kmeans_errors)
    # print("Error rate (%):", (kmeans_errors / total_pairs) * 100)
    #
    # random_assignments = random_baseline_clustering(X_pca_normalized, 7)
    # random_errors , _ = count_errors(random_assignments, y_subset)
    # print("Number of errors made by Random Baseline on PCA-reduced data:", random_errors)
    # print("Error rate (%):", (random_errors / total_pairs) * 100)
    #
    # # Step 7: Visualize the clustering results (optional)
    # print("Visualizing K-means clustering results...")
    # plt.figure(figsize=(8, 8))
    # plt.scatter(X_pca_normalized[:, 0], X_pca_normalized[:, 1], c=kmeans_assignments, cmap='viridis', alpha=0.8)
    # plt.title("K-means Clustering on PCA-Reduced Data")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
