import numpy as np
from kernels import linear, polynomial, rbf, sigmoid
import quadprog

class SVC:
    def __init__(self, C=1.0, kernel='linear', gamma=0.1):
        self.coef_ = None
        self.intercept_ = None
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def _kernel(self, X1, X2):
        if self.kernel == 'linear':
            return linear(X1, X2)

        elif self.kernel == 'polynomial':
            return polynomial(X1, X2)

        elif self.kernel == 'rbf':
            return rbf(X1, X2, self.gamma)

        elif self.kernel == 'sigmoid':
            return sigmoid(X1, X2)

        else:
            raise ValueError('Invalid kernel type')

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(float)
        y = np.where(y == 0, -1, y)

        n = X.shape[0]
        K = self._kernel(X, X)

        P = np.outer(y, y) * K
        q = -np.ones(n)
        G = P + 1e-8 * np.eye(n)
        a = q

        C_mat = y.reshape(1, -1)
        b = np.array([0.0])
        lb = np.zeros(n)
        ub = np.ones(n) * self.C
        alpha = quadprog.solve_qp(G, a, C_mat, b, 1, lb, ub)[0]
        self.coef_ = alpha

        # support vectors
        sv = alpha > 1e-5
        self.X_sv = X[sv]
        self.y_sv = y[sv]
        self.alpha_sv = alpha[sv]

        # bias term
        self.intercept_ = 0
        sv_idx = np.where(sv)[0]

        for i in range(len(sv_idx)):
            idx_i = sv_idx[i]

            self.intercept_ += self.y_sv[i] - np.sum(self.alpha_sv * self.y_sv * K[sv_idx, idx_i])

        self.intercept_ /= len(sv_idx)

        return self


    def predict(self, X):
        X = np.asarray(X)

        K = self._kernel(X, self.X_sv)

        decision = np.sum(self.alpha_sv * self.y_sv * K,axis=1) + self.intercept_

        return np.sign(decision)
