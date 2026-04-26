import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

        self.class_count_ = None
        self.class_log_prior_ = None
        self.feature_count_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.class_count_ = np.zeros(n_classes)

        self.feature_count_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[i] = X_c.shape[0]
            self.feature_count_[i] = X_c.sum(axis=0)

        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_ / n_samples)

        else: self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_sum = smoothed_fc.sum(axis=1, keepdims=True)

        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_sum)

        return self

    def predict(self, X):
        X = np.array(X)

        if self.class_log_prior_ is None:
            raise ValueError("Model not fitted yet")

        log_probs = self.class_log_prior_ + X @ self.feature_log_prob_.T

        return self.classes_[np.argmax(log_probs, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
