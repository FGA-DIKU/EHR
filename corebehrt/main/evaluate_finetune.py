import logging
from os.path import join
import pandas as pd
import torch
import os 

from corebehrt.constants.paths import FOLDS_FILE, PREPARED_ALL_PATIENTS
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.evaluate_finetune import inference_fold, compute_metrics
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.config import instantiate_function

CONFIG_PATH = "./corebehrt/configs/finetune_evaluate.yaml"


def main_evaluate(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate()

    # Logger
    logger = logging.getLogger("evaluate")

    # Setup config
    cfg.trainer_args = {}
    batch_size_value = cfg.get("test_batch_size", 128)
    for key in ["test_batch_size", "effective_batch_size", "batch_size"]:
        cfg.trainer_args[key] = batch_size_value
    cfg.paths.restart_model = cfg.paths.model

    # Load data
    loaded_data = torch.load(
        join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS), weights_only=False
    )
    test_data = PatientDataset(loaded_data)
    test_dataset = BinaryOutcomeDataset(test_data.patients)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.folds_dir, FOLDS_FILE), weights_only=False)
    check_for_overlap(folds, test_pids, logger)
    targets = test_data.get_outcomes()

    # Get predictions
    combined_df = pd.DataFrame(
        {
            "pid": test_pids,
            "target": targets,
        }
    )
    if cfg.get("save_info", False):
        for k, v in cfg.save_info.items():
            func = instantiate_function(v)
            combined_df[k] = func(test_dataset)

    all_probas = []
    for n_fold, fold in enumerate(folds, start=1):
        probas, embeddings = inference_fold(
            finetune_folder=cfg.paths.model,
            cfg=cfg,
            test_data=test_dataset,
            logger=logger,
            fold=n_fold,
        )
        combined_df[f"fold_{n_fold}_probas"] = probas
        all_probas.append(probas)

        # Save embeddings if specified
        if embeddings is not None:
            save_emb_path = join(cfg.paths.predictions, "embeddings", f"fold_{n_fold}")
            os.makedirs(save_emb_path, exist_ok=True)
            torch.save(embeddings[0], join(save_emb_path, "ehr_embeddings.pt"))
            torch.save(embeddings[1], join(save_emb_path, "BERT_embeddings.pt"))
            torch.save(embeddings[2], join(save_emb_path, "cls_embeddings.pt"))
            torch.save(test_pids, join(save_emb_path, "pids.pt"))

    # Save combined predictions and metrics if specified
    combined_df.to_csv(join(cfg.paths.predictions, "predictions.csv"), index=False)
    compute_metrics(cfg, targets, all_probas, logger)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
