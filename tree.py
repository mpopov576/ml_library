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
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='gini', max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        if criterion not in ['gini', 'entropy']:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.criterion = criterion
        self.root = None
        self.max_features = max_features

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

        n_features = X.shape[1]
        features = np.arange(n_features)

        if self.max_features is not None:
            features = np.random.choice(features, self.max_features, replace=False)

        for feature in features:
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



class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))

        elif self.max_features == 'log2':
            return int(np.log2(n_features))

        elif isinstance(self.max_features, int):
            return self.max_features

        else:
            return n_features

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        max_features = self._get_max_features(n_features)

        self.trees = []

        for _ in range(self.n_estimators):
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf,
                                          max_features=max_features,)

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.array(X)

        all_preds = np.array([tree.predict(X) for tree in self.trees])

        final_preds = []
        for i in range(X.shape[0]):
            counts = np.bincount(all_preds[:, i])
            final_preds.append(np.argmax(counts))

        return np.array(final_preds)

    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)

        preds = self.predict(X)
        return np.mean(preds == y)



class AdaBoostClassifier:
    def __init__(self, n_estimators, lr=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.lr = lr
        self.random_state = random_state

        self.models = []
        self.alphas = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.where(np.array(y) == 0, -1, 1)

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        sample_weights = np.ones(n_samples) / n_samples

        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=sample_weights)

            X_sample = X[indices]
            y_sample = y[indices]

            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X_sample, y_sample)

            preds = model.predict(X)

            incorrect = (preds != y)
            err = np.sum(sample_weights * incorrect)

            err = max(err, 1e-10)

            alpha = self.lr * 0.5 * np.log((1 - err) / err)

            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)

            self.models.append(model)
            self.alphas.append(alpha)

        return self

    def predict(self, X):
        X = np.array(X)

        final = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            preds = model.predict(X)
            final += alpha * preds

        return np.where(final >= 0, 1, -1)

    def score(self, X, y):
        X = np.array(X)
        y = np.where(np.array(y) == 0, -1, 1)

        return np.mean(y == self.predict(X))
