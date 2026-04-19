

def accuracy_score(y_true, y_pred, normalize=True):
    if normalize:
        return (y_true==y_pred).sum()/y_true.size

    return (y_true==y_pred).sum()
