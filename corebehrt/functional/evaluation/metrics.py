from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

def PR_AUC(probas, targets):
    try:
        return average_precision_score(targets, probas)
    except:
        logger.warn("PR AUC score could not be computed")
        return 0

def ROC_AUC(probas, targets):
    try:
        return roc_auc_score(targets, probas)
    except:
        logger.warn("ROC AUC score could not be computed")
        return 0