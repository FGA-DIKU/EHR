import logging

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)  # Get the logger for this module

"""Computes the precision@k for the specified value of k"""


class PrecisionAtK:
    def __init__(self, topk=10):
        """Computes the precision@k for the specified value of k"""
        self.topk = topk

    def __call__(self, outputs, batch):
        logits = outputs.logits
        labels = outputs.labels

        _, pred = logits.topk(self.topk, -1, True, True)
        pred = pred.t()
        correct = pred.eq(labels)
        if correct.numel() == 0:
            return 0
        else:
            return correct.any(0).float().mean().item()


class LossAccessor:
    def __init__(self, loss_name):
        self.loss_name = loss_name

    def __call__(self, outputs, batch):
        return outputs.__getattribute__(self.loss_name).cpu()

class Perplexity:
    """Compute perplexity for language modeling."""
    
    def __call__(self, outputs, batch):
        # Perplexity = exp(cross_entropy_loss)
        loss = outputs.loss
        return torch.exp(loss).item()

class TokenAccuracy:
    """Compute token-level accuracy for next token prediction."""
    
    def __call__(self, outputs, batch):
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        labels = outputs.labels  # Shape: (batch_size, seq_len)
        
        # Handle case where logits and labels have different lengths due to shifting
        if logits.shape[1] != labels.shape[1]:
            # Logits are shifted (remove last position), labels are not shifted
            logits = logits[:, :-1, :]  # Remove last position from logits
        
        # Get predicted tokens
        pred_tokens = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)
        
        # Calculate accuracy
        correct = (pred_tokens == labels).float()
        # Only consider non-padded tokens
        mask = (labels != -100).float()  # -100 is padding for loss calculation
        if mask.sum() == 0:
            return 0.0
        
        accuracy = (correct * mask).sum() / mask.sum()
        return accuracy.item()


class NextTokenPrecisionAtK:
    """Compute precision@k for next token prediction."""
    
    def __init__(self, topk=10):
        self.topk = topk
    
    def __call__(self, outputs, batch):
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        labels = outputs.labels  # Shape: (batch_size, seq_len)
        
        # Handle case where logits and labels have different lengths due to shifting
        if logits.shape[1] != labels.shape[1]:
            # Logits are shifted (remove last position), labels are not shifted
            # We need to align them by removing the last position from logits
            logits = logits[:, :-1, :]  # Remove last position from logits
        
        # Get top-k predictions for each position
        _, pred_indices = logits.topk(self.topk, dim=-1)  # Shape: (batch_size, seq_len, k)
        
        # Check if true label is in top-k predictions
        correct = torch.zeros_like(labels, dtype=torch.bool)
        for i in range(self.topk):
            correct |= (pred_indices[:, :, i] == labels)
        
        # Only consider non-padded tokens
        mask = (labels != -100)
        if mask.sum() == 0:
            return 0.0
        
        precision = (correct & mask).float().sum() / mask.float().sum()
        return precision.item()


def binary_hit(outputs, batch, threshold=0.5, average=True):
    logits = outputs.logits
    target = batch["target"]

    probs = torch.sigmoid(logits)
    predictions = (probs > threshold).long().view(-1)  # TODO: Add uncertainty measure

    if not average:
        return (predictions == target).float()

    else:
        return (predictions == target).float().mean().item()


class BaseMetric:
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold

    def _return_probas_and_targets(self, outputs, batch):
        probas = torch.sigmoid(outputs.logits)
        return probas.cpu(), batch["target"].cpu()

    def _return_predictions_and_targets(self, outputs, batch):
        probas, targets = self._return_probas_and_targets(outputs, batch)
        predictions = (probas > self.threshold).long().view(-1)
        return predictions, targets

    def _return_confusion_matrix(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targets(outputs, batch)
        return confusion_matrix(targets, predictions, labels=[0, 1]).ravel()

    def __call__(self, outputs, batch):
        raise NotImplementedError


class Accuracy(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targets(outputs, batch)
        try:
            return accuracy_score(targets, predictions)
        except:
            logger.warn("Accuracy score could not be computed")
            return 0


class Dice(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return (2.0 * tp) / (2 * tp + fp + fn)


class Balanced_Accuracy(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targets(outputs, batch)
        try:
            return balanced_accuracy_score(targets, predictions)
        except:
            logger.warn("Balanced accuracy score could not be computed")
            return 0


class Precision(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targets(outputs, batch)
        return precision_score(targets, predictions, zero_division=0)


class Recall(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targets(outputs, batch)
        return recall_score(targets, predictions, zero_division=0)


class ROC_AUC(BaseMetric):
    def __call__(self, outputs, batch):
        probas, targets = self._return_probas_and_targets(outputs, batch)
        try:
            return roc_auc_score(targets, probas)
        except:
            logger.warn("ROC AUC score could not be computed")
            return 0


class PR_AUC(BaseMetric):
    def __call__(self, outputs, batch):
        probas, targets = self._return_probas_and_targets(outputs, batch)
        try:
            return average_precision_score(targets, probas)
        except:
            logger.warn("PR AUC score could not be computed")
            return 0


class F1(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targets(outputs, batch)
        return f1_score(targets, predictions, zero_division=0)


class Cohen_Kappa(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targets(outputs, batch)
        return cohen_kappa_score(targets, predictions)


class Matthews_Correlation_Coefficient(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targets(outputs, batch)
        return matthews_corrcoef(targets, predictions)


class Percentage_Positives(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, _ = self._return_predictions_and_targets(outputs, batch)
        return predictions.float().mean().item()


class Mean_Probability(BaseMetric):
    def __call__(self, outputs, batch):
        probas, _ = self._return_probas_and_targets(outputs, batch)
        return probas.mean().item()


class True_Positives(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return tp


class False_Positives(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return fp


class True_Negatives(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return tn


class False_Negatives(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return fn


def specificity(y, y_scores):
    tn, fp, fn, tp = confusion_matrix(y, y_scores).ravel()
    return tn / (tn + fp)
