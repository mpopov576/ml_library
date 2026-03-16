import numpy as np


def train_test_split(X,
                     y,
                     test_size=0.25,
                     train_size=None,
                     shuffle=True,
                     random_state=42,
                     stratify=None):
    if train_size is None:
        train_size = 1 - test_size

    if stratify is None:
        groups = None
    else:
        groups = stratify

    if shuffle:
        rng = np.random.default_rng(random_state)

        if groups is not None:
            unique_classes = groups.unique()
            train_indices = []
            test_indices = []
            for c in unique_classes:
                c_idx = groups[groups == c].index.values
                rng.shuffle(c_idx)
                split_point = int(len(c_idx) * train_size)
                train_indices.extend(c_idx[:split_point])
                test_indices.extend(c_idx[split_point:])

        else:
            indices = np.arange(len(X))
            rng.shuffle(indices)
            split_point = int(len(indices) * train_size)
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
    else:
        indices = np.arange(len(X))
        split_point = int(len(indices) * train_size)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    X_test = X.iloc[test_indices]

    return X_train, X_test, y_train, y_test
