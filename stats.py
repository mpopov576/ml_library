import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(X):
    X=np.asarray(X)

    # for numerical stability
    X_shifted = X - np.max(X, axis=1, keepdims=True)

    return np.exp(X_shifted) / np.sum(np.exp(X_shifted), axis=1, keepdims=True)

def gini_index(y):
    y = np.asarray(y)

    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    return float(1.0 - np.sum(probs ** 2))

def entropy(y):
    y = np.asarray(y)

    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    probs = probs[probs > 0]

    return float(-np.sum(probs * np.log2(probs)))
