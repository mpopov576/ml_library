import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(X):
    X=np.asarray(X)

    #for numerical stability
    X_shifted = X - np.max(X, axis=1, keepdims=True)

    return np.exp(X_shifted) / np.sum(np.exp(X_shifted), axis=1, keepdims=True)
