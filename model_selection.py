import math
import numpy as np

def train_test_split(X, y, test_size=0.25, train_size=0.75, shuffle=True, random_state=None, stratify=None):
    if not math.isclose(train_size + test_size, 1):
        raise ValueError("train_size and test_size must sum up to 1")

    size_X = X.shape[0]
    size_y = y.shape[0]

    if size_X != size_y:
        raise ValueError("X and y must have the same number of rows")

    if stratify is not None:
        rng=np.random.default_rng(random_state)
        train_idx = []
        test_idx = []

        for label in np.unique(stratify):
            label_indices = np.where(stratify==label)[0]
            rng.shuffle(label_indices)

            split = int(len(label_indices) * train_size)
            train_idx.extend(label_indices[:split])
            test_idx.extend(label_indices[split:])

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        return X_train, X_test, y_train, y_test


    elif shuffle:
        rng = np.random.default_rng(random_state)
        indices = np.arange(size_X)
        rng.shuffle(indices)
        X = X[indices]
        y = y[indices]

    X_train = X[:(int)(train_size * size_X)]
    y_train = y[:(int)(train_size * size_y)]
    X_test = X[(int)(train_size * size_X):]
    y_test = y[(int)(train_size * size_y):]

    return X_train, X_test, y_train, y_test

