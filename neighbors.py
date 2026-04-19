from metrics import accuracy_score
import numpy as np

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        # use weighted vote with inversing the distance: 1/(d+eps) (eps is to prevent division by 0)
        # eps is small so weights differ more based on the distance
        eps = 1e-9
        y_pred = []

        for x in X:
            if self.metric == 'euclidean':
                distances = np.sqrt(np.sum((x - self.X) ** 2, axis=1))

            elif self.metric == 'manhattan':
                distances = np.sum(np.abs(x - self.X), axis=1)

            else: raise ValueError("Unknown metric")

            idx = np.argsort(distances)[:self.n_neighbors]

            k_labels = self.y[idx]
            k_distances = distances[idx]
  
            weights = 1 / (k_distances + eps)

            votes = {}
            for label, weight in zip(k_labels, weights):
                votes[label] = votes.get(label, 0) + weight

            y_pred.append(max(votes, key=votes.get))

        return np.array(y_pred)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


