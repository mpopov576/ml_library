import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[indices]

    def _compute_distances(self, X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def fit(self, X):
        X = np.array(X)

        centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in range(self.n_clusters)])

            if np.allclose(new_centroids, centroids):
                break

            centroids = new_centroids

        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)

        inertia = np.sum((X - centroids[labels]) ** 2)

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia

        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError('Must fit before predicting')

        X = np.array(X)
        distances = self._compute_distances(X, self.cluster_centers_)

        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
