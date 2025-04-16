from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

def ROC_AUC(y_true, y_pred):
    """
    Calculate the ROC AUC score.

    Parameters:
    y_true (list): True binary labels.
    y_pred (list): Target scores.

    Returns:
    float: ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)

def PR_AUC(y_true, y_pred):
    """
    Calculate the average precision score.

    Parameters:
    y_true (list): True binary labels.
    y_pred (list): Target scores.

    Returns:
    float: Average precision score.
    """
    return average_precision_score(y_true, y_pred)