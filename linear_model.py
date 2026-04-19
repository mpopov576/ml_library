import numpy as np
from metrics import r2_score

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if np.allclose(y, y.mean()):
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = y.mean()
            return self

        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise ValueError("Matrix is not invertible. Please remove collinear features.")

        # normal equation is used instead of gd
        self.coef_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept_ = y.mean() - X.mean(axis=0).dot(self.coef_)

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))

