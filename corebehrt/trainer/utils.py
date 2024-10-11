import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_tqdm(dataloader: DataLoader) -> tqdm:
    return tqdm(dataloader, total=len(dataloader))


def compute_avg_metrics(metric_values: dict):
    """Computes the average of the metric values when metric is not zero and not NaN"""
    averages = {}
    for name, values in metric_values.items():
        values_array = np.array(values)
        select_mask = (values_array == 0) | (np.isnan(values_array))
        non_zero_values = values_array[~select_mask]

        if non_zero_values.size:
            averages[name] = np.mean(non_zero_values)
        else:
            averages[name] = 0
    return averages


def save_curves(
    run_folder: str, logits: torch.Tensor, targets: torch.Tensor, epoch: int, mode="val"
) -> None:
    """Saves the ROC and PRC curves to a csv file in the run folder"""
    roc_name = os.path.join(run_folder, "checkpoints", f"roc_curve_{mode}_{epoch}.npz")
    prc_name = os.path.join(run_folder, "checkpoints", f"prc_curve_{mode}_{epoch}.npz")
    probas = torch.sigmoid(logits).cpu().numpy()
    fpr, tpr, threshold_roc = roc_curve(targets, probas)
    precision, recall, threshold_pr = precision_recall_curve(targets, probas)
    np.savez_compressed(roc_name, fpr=fpr, tpr=tpr, threshold=threshold_roc)
    np.savez_compressed(
        prc_name,
        precision=precision,
        recall=recall,
        threshold=np.append(threshold_pr, 1),
    )


def save_predictions(
    run_folder: str, logits: torch.Tensor, targets: torch.Tensor, epoch: int, mode="val"
) -> None:
    """Saves the predictions to npz files in the run folder"""
    probas_name = os.path.join(run_folder, "checkpoints", f"probas_{mode}_{epoch}.npz")
    targets_name = os.path.join(
        run_folder, "checkpoints", f"targets_{mode}_{epoch}.npz"
    )
    probas = torch.sigmoid(logits).cpu().numpy()
    np.savez_compressed(probas_name, probas=probas)
    np.savez_compressed(targets_name, targets=targets)


def save_metrics_to_csv(run_folder: str, metrics: dict, epoch: int, mode="val") -> None:
    """Saves the metrics to a csv file"""
    metrics_name = os.path.join(run_folder, "checkpoints", f"{mode}_scores_{epoch}.csv")
    with open(metrics_name, "w") as file:
        file.write("metric,value\n")
        for key, value in metrics.items():
            file.write(f"{key},{value}\n")
