import numpy as np
from metrics import r2_score, accuracy_score
from stats import sigmoid

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if np.allclose(y, y.mean()):
            self.fitted = True
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = y.mean()
            return self

        if np.linalg.matrix_rank(X.T @ X) < (X.T @ X).shape[1]:
            raise ValueError("Matrix is not invertible. Please remove collinear features.")

        # normal equation is used instead of gd
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = y.mean() - X.mean(axis=0) @ self.coef_
        self.fitted = True
        return self

    def predict(self, X):
        if self.fitted:
            return X @ self.coef_ + self.intercept_

        raise ValueError("The model has not been fitted yet.")

    def score(self, X, y):
        return r2_score(y, self.predict(X))



class LogisticRegression:
    def __init__(self, C=1.0, lr=0.01, max_iter=1000, random_state=42):
        self.coef_ = None
        self.intercept_ = None
        self.C = C
        self.random_state = random_state
        self.fitted = False
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        self.coef_ = np.random.rand(n_classes, n_features) * 0.01
        self.intercept_ = np.zeros(n_classes)

        for _ in range(self.max_iter):
            for i, cls in enumerate(classes):
                y_bin = (y == cls).astype(float)
                logits = X @ self.coef_[i] + self.intercept_[i]
                p = sigmoid(logits)

                error = p - y_bin
                self.coef_[i] -= self.lr * ((X.T @ error) / n_samples + (1 / self.C) * self.coef_[i])
                self.intercept_[i] -= self.lr * np.mean(error)

        self.fitted = True
        return self

    def predict_proba(self, X):
        if self.fitted:
            X = np.asarray(X)
            logits = X @ self.coef_.T + self.intercept_
            probs = sigmoid(logits)
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-15, None)
            return probs

        raise ValueError("The model has not been fitted yet.")

    def predict(self, X):
        if self.fitted:
            return np.argmax(self.predict_proba(X), axis=1)

        raise ValueError("The model has not been fitted yet.")

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
