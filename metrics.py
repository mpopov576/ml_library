import numpy as np

def accuracy_score(y_true, y_pred, normalize=True):
    if normalize:
        return (y_true==y_pred).sum()/y_true.size

    return (y_true==y_pred).sum()

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a-b))
