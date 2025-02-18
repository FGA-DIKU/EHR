import os
from datetime import datetime
from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from corebehrt.azure.log import setup_metrics_dir, log_metric


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
    targets = targets.cpu().numpy()
    np.savez_compressed(probas_name, probas=probas)
    np.savez_compressed(targets_name, targets=targets)


def save_metrics_to_csv(run_folder: str, metrics: dict, epoch: int, mode="val") -> None:
    """Saves the metrics to a csv file"""
    metrics_name = os.path.join(run_folder, "checkpoints", f"{mode}_scores_{epoch}.csv")
    with open(metrics_name, "w") as file:
        file.write("metric,value\n")
        for key, value in metrics.items():
            file.write(f"{key},{value}\n")


def compute_and_save_scores_mean_std(
    n_splits: int, finetune_folder: str, mode="val"
) -> None:
    """Compute mean and std of test/val scores. And save to finetune folder."""
    scores = []
    for fold in range(1, n_splits + 1):
        fold_checkpoints_folder = join(finetune_folder, f"fold_{fold}", "checkpoints")
        last_epoch = max(
            [
                int(f.split("_")[-2].split("epoch")[-1])
                for f in os.listdir(fold_checkpoints_folder)
                if f.startswith("checkpoint_epoch")
            ]
        )
        table_path = join(fold_checkpoints_folder, f"{mode}_scores_{last_epoch}.csv")
        if not os.path.exists(table_path):
            continue
        fold_scores = pd.read_csv(
            join(fold_checkpoints_folder, f"{mode}_scores_{last_epoch}.csv")
        )
        scores.append(fold_scores)
    scores = pd.concat(scores)
    scores_mean_std = scores.groupby("metric")["value"].agg(["mean", "std"])
    date = datetime.now().strftime("%Y%m%d-%H%M")
    scores_mean_std.to_csv(join(finetune_folder, f"{mode}_scores_mean_std_{date}"))

    # Log to Azure
    with setup_metrics_dir(f"{mode} scores"):
        for idx, row in scores_mean_std.iterrows():
            for col in scores_mean_std.columns:
                log_metric(f"{idx} {col}", row[col])
