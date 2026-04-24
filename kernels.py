import numpy as np

def linear(X, Y=None):
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X @ Y.T

def polynomial(X, Y=None, degree=3, gamma=1.0, coef0=1.0):
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    return ((gamma * X @ Y.T) + coef0) ** degree

def rbf(X, Y=None, gamma=1.0):
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    X_norm = np.sum(X**2, axis=1)[:, None]
    Y_norm = np.sum(Y**2, axis=1)[None, :]

    K = X_norm + Y_norm - 2 * (X @ Y.T)
    K = np.maximum(K, 0.0)

    return np.exp(-gamma * K)

def sigmoid(X, Y=None, gamma=1.0, coef0=1.0):
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    return np.tanh(gamma * (X @ Y.T) + coef0)
