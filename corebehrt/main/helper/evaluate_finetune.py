from os.path import join
from typing import List
import torch
import pandas as pd

from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.inference import EHRInferenceRunner
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset
from corebehrt.modules.setup.config import instantiate_function


def inference_fold(
    finetune_folder: str,
    cfg: dict,
    test_data: BinaryOutcomeDataset,
    logger,
    fold: int,
    shap_dict: dict,
) -> None:
    fold_folder = join(finetune_folder, f"fold_{fold}")

    # Load model
    modelmanager_trained = ModelManager(cfg, fold)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_finetune_model(checkpoint, [])
    print(f"Model loaded from {fold_folder}")

    # Run inference
    return_embeddings = cfg.get("return_embeddings", False)
    evaluater = EHRInferenceRunner(
        model=model,
        test_dataset=test_data,  # test only after training
        args=cfg.trainer_args,
        cfg=cfg,
    )
    logits_tensor, targets_tensor, embeddings_tensor, shap_values = evaluater.inference_loop(
        return_embeddings=return_embeddings,
        shap_dict=shap_dict
    )
    probas = torch.sigmoid(logits_tensor.detach()).numpy()

    return probas, embeddings_tensor, shap_values


def get_sequence_length(dataset: BinaryOutcomeDataset) -> List[int]:
    lengths = [len(patient.concepts) for patient in dataset.patients]
    return lengths


def compute_metrics(cfg, targets, all_probas, logger):
    """
    Computes and saves metrics for each fold and the average metrics.
    """
    if not hasattr(cfg, "metrics") or not cfg.metrics:
        return

    metrics = {k: instantiate_function(v) for k, v in cfg.metrics.items()}
    fold_metrics_list = []

    for n_fold, probas in enumerate(all_probas, start=1):
        fold_metrics = {name: func(targets, probas) for name, func in metrics.items()}
        fold_metrics["fold"] = f"fold_{n_fold}"
        fold_metrics_list.append(fold_metrics)

    metrics_df = pd.DataFrame(fold_metrics_list)
    avg_metrics = metrics_df.drop(columns=["fold"]).mean().to_dict()
    avg_metrics["fold"] = "average"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    metrics_df.to_csv(join(cfg.paths.predictions, "metrics.csv"), index=False)

    logger.info("Average metrics:")
    for key in avg_metrics:
        if key != "fold":
            logger.info(f"{key}: {avg_metrics[key]:.4f}")
