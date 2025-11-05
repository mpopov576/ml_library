import numpy as np

def accuracy_score(y_true, y_pred, normalize=True):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_shape must have the same shape')

    correct = np.sum(y_pred == y_true)

    if normalize:
        return correct / len(y_true)
    else:
        return correct


def euclidean_distance(x, y):
    p1 = np.array(x)
    p2 = np.array(y)
    if p1.shape != p2.shape:
        raise ValueError('shape of x and y must be equal')

    return np.sqrt(np.sum(np.power(p1 - p2, 2)))
