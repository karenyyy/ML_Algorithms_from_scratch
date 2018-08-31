import numpy as np
from utils.util import normalize, euclidean_distance


class KMeans(object):
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations

    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            # randomly select k centroids
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def closest_to_which_centroid(self, sample, centroids):
        # assert centroids.shape[0] == self.k
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            dist = euclidean_distance(sample, centroid)
            if dist < closest_dist:
                closest_i = i
                closest_dist = dist
        return closest_i

    def clusterize(self, centroids, X):
        # assert centroids.shape[0] == self.k
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self.closest_to_which_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(X.shape[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        centroids = self.init_random_centroids(X)
        for _ in range(self.max_iterations):
            clusters = self.clusterize(centroids, X)
            prev_centroids = centroids
            centroids = self.update_centroids(clusters, X)
            diff = centroids - prev_centroids
            if not diff.any():
                break
            return self.get_cluster_labels(clusters, X)
