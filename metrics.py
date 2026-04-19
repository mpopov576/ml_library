import numpy as np

def accuracy_score(y_true, y_pred, normalize=True):
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
    errm =  np.sum((y_true - np.mean(y_true))**2)
    errp = np.sum((y_true-y_pred)**2)
    if errp == 0 and errm == 0 :
        return 1
    elif errm  == 0:
        return 0
    return 1-errp/errm
    
