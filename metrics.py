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


def manhattan_distance(x, y):
    p1 = np.array(x)
    p2 = np.array(y)

    if p1.shape != p2.shape:
        raise ValueError("shape of x and y must be equal")

    return np.sum(np.abs(p1 - p2))

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)

    ss_tot = np.sum((y_true - y_mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - ss_res / ss_tot

def root_mean_squared_error(y_true, y_pred):
    n = y_true.shape[0]
    s_er = np.sum((y_true - y_pred)**2)

    return np.sqrt(s_er / n)
