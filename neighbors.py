import numpy as np
from ml_training.ml_lib import metrics


class KNeighborsClassifier:

    def __init__(self, n_neighbors=5, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def compute_distance(self, x):
        if self.metric == "euclidean":
            return np.array(
                [metrics.euclidean_distance(x, xi) for xi in self.X_train])
        elif self.metric == "manhattan":
            return np.array(
                [metrics.manhattan_distance(x, xi) for xi in self.X_train])
        else:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x in X:
            distances = self.compute_distance(x)
            neighbor_idxes = distances.argsort()[:self.n_neighbors]
            neighbor_labels = self.y_train[neighbor_idxes]

            values, counts = np.unique(neighbor_labels, return_counts=True)
            predictions.append(values[counts.argmax()])

        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return metrics.accuracy_score(y, y_pred)
