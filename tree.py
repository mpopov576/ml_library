import numpy as np
from stats import gini_index, entropy

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        if criterion not in ['gini', 'entropy']:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.criterion = criterion
        self.root = None

    def _impurity(self, y):
        return gini_index(y) if self.criterion == 'gini' else entropy(y)

    def _majority_class(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        n = len(y)
        parent_impurity = self._impurity(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf: continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                child_impurity = len(y_left) / n * self._impurity(y_left) + len(y_right) / n * self._impurity(y_right)

                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        n_samples = len(y)

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=self._majority_class(y))

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return Node(value=self._majority_class(y))

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.root = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)

        return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.array(X)

        return np.array([self._predict_one(x, self.root) for x in X])
