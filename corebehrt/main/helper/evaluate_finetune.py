import os
from os.path import join
from typing import List

import torch

from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.inference import EHRInferenceRunner
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset



def evaluate_fold(
    finetune_folder: str,
    cfg: dict,
    test_data: BinaryOutcomeDataset,
    logger,
    fold: int,
) -> None:
    fold_folder = join(finetune_folder, f"fold_{fold}")

    # Load model
    modelmanager_trained = ModelManager(cfg, fold)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_finetune_model(checkpoint, [])
    print(f"Model loaded from {fold_folder}")

    # Run inference
    evaluater = EHRInferenceRunner(
        model=model,
        test_dataset=test_data,  # test only after training
        args=cfg.trainer_args,
        cfg=cfg,
    )
    logits_tensor, targets_tensor = evaluater.inference_loop()    
    probas = torch.sigmoid(logits_tensor).numpy()

    return probas

def get_sequence_length(dataset:BinaryOutcomeDataset) -> List[int]:
    lengths = [len(patient.concepts) for patient in dataset.patients]
    return lengths