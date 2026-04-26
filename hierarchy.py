import numpy as np

def _euclidean(a, b):
    return np.linalg.norm(a - b)

def linkage(X, method='single'):
    X = np.array(X)
    n = X.shape[0]

    clusters = {i: [i] for i in range(n)}
    centroids = {i: X[i].copy() for i in range(n)}

    active = set(range(n))
    Z = []
    next_id = n

    def cluster_distance(c1, c2):
        pts1 = clusters[c1]
        pts2 = clusters[c2]

        if method == 'single':
            return min(_euclidean(X[i], X[j]) for i in pts1 for j in pts2)

        elif method == 'complete':
            return max(_euclidean(X[i], X[j]) for i in pts1 for j in pts2)

        elif method == 'average':
            dists = [_euclidean(X[i], X[j]) for i in pts1 for j in pts2]
            return np.mean(dists)

        elif method == 'centroid':
            return _euclidean(centroids[c1], centroids[c2])

        else:
            raise ValueError('Unknown method')

    while len(active) > 1:
        active_list = list(active)

        best_pair = None
        best_distance = float('inf')

        for i in range(len(active_list)):
            for j in range(i+1, len(active_list)):
                c1 = active_list[i]
                c2 = active_list[j]

                dist = cluster_distance(c1, c2)

                if dist < best_distance:
                    best_distance = dist
                    best_pair = (c1, c2)

        c1, c2 = best_pair

        new_id = next_id
        next_id += 1

        clusters[new_id] = clusters[c1] + clusters[c2]

        if method == 'centroid':
            centroids[new_id] = np.mean(X[clusters[new_id]], axis=0)

        size = len(clusters[new_id])

        Z.append([c1, c2, best_distance, size])

        active.remove(c1)
        active.remove(c2)
        active.add(new_id)

    return np.array(Z)

class AgglomerativeHierarchy:
    def __init__(self, n_clusters=2, linkage_method='single'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.labels_ = None

    def fit(self, X):
        Z = linkage(X, method=self.linkage_method)
        n = X.shape[0]

        clusters = {i: [i] for i in range(n)}

        next_id = n

        for row in Z:
            c1, c2, distance, size = row
            c1, c2 = int(c1), int(c2)

            new_cluster = clusters[c1] + clusters[c2]

            del clusters[c1]
            del clusters[c2]

            clusters[next_id] = new_cluster
            next_id += 1

            if len(clusters) == self.n_clusters:
                break

        self.labels_ = np.zeros(n, dtype=int)

        for label, pts in enumerate(clusters.values()):
            for pt in pts:
                self.labels_[pt] = label

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_







