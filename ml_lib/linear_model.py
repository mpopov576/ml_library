import numpy as np
from ml_lib.metrics import r2_score


class LinearRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)

        X_b = np.hstack((np.ones((X.shape[0], 1)), X))

        XtX = X_b.T @ X_b

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Matrix is not invertible. Please remove collinear features.")

        beta = XtX_inv @ X_b.T @ y

        self.intercept_ = beta[0, 0]
        self.coef_ = beta[1:, 0]

        return self

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return (X @ self.coef_ + self.intercept_).flatten()

    def score(self, X, y):
        return r2_score(y, self.predict(X))
