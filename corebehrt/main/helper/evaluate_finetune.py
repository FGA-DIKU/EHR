import os
from os.path import join
from typing import List

import torch

from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.evaluater import EHREvaluator
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset



def evaluate_fold(
    finetune_folder: str,
    finetune_cfg: dict,
    test_data: BinaryOutcomeDataset,
    logger,
    fold: int,
    return_preds: bool,
    metrics: dict = {},
) -> None:
    fold_folder = join(finetune_folder, f"fold_{fold}")

    # Load model
    modelmanager_trained = ModelManager(finetune_cfg, fold)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_finetune_model(checkpoint)
    
    # Run inference
    evaluater = EHREvaluator(
        model=model,
        test_dataset=test_data,  # test only after training
        args=finetune_cfg.trainer_args,
        cfg=finetune_cfg,
    )
    logits_tensor, targets_tensor = evaluater.inference_loop()

    # Compute metrics
    logger.info(f"Computing metrics for fold {fold}")
    computed_metrics = {}
    for name, func in metrics.items():
        v = func(outputs, batch)
        logger.info(f"{name}: {v}")
        computed_metrics[name] = v

    if return_preds:
        probas = torch.sigmoid(logits_tensor).numpy()
        return probas, computed_metrics
    else:
        return None, computed_metrics


def get_sequence_length(dataset:BinaryOutcomeDataset) -> List[int]:
    lengths = [len(patient.concepts) for patient in dataset.patients]
    return lengths