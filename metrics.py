import numpy as np

def accuracy_score(y_true, y_pred, normalize=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if normalize:
        return (y_true==y_pred).sum()/y_true.size

    return (y_true==y_pred).sum()

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a-b))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errm = np.sum((y_true - np.mean(y_true))**2)
    errp = np.sum((y_true-y_pred)**2)

    if errp == 0 and errm == 0 :
        return 1
    elif errm  == 0:
        return 0
    return 1-errp/errm

def root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.sum((y_true-y_pred)**2)/len(y_true))

def recall_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))

        if tp + fn == 0:
            recalls.append(0.0)
        else: recalls.append(tp / (tp + fn))

    return np.mean(recalls)


def precision_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    precisions = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))

        if tp + fp == 0:
            precisions.append(0.0)
        else: precisions.append(tp / (tp + fp))

    return np.mean(precisions)

def f1_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    if p + r == 0:
        return 0

    return 2*p*r/(p+r)



