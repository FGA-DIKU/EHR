import logging
import os
from os.path import join

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from corebehrt.constants.paths import (
    FOLDS_FILE,
    PREPARED_ALL_PATIENTS,
    FINETUNE_CFG
)
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.evaluate_finetune import evaluate_fold
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.modules.monitoring.metric_aggregation import (
    compute_and_save_scores_mean_std,
)
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.config import instantiate_function
import pandas as pd

CONFIG_PATH = "./corebehrt/configs/finetune_evaluate.yaml"


def main_evaluate(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate()

    # Logger
    logger = logging.getLogger("evaluate")

    # Setup config and metrics
    cfg.trainer_args = {}
    cfg.trainer_args.test_batch_size = cfg.get("test_batch_size", 128)
    cfg.trainer_args.effective_batch_size = cfg.get("test_batch_size", 128)
    cfg.trainer_args.batch_size = cfg.get("test_batch_size", 128)
    cfg.paths.restart_model = cfg.paths.model
    # metrics = {k: instantiate_function(v) for k, v in cfg.metrics.items()} if cfg.metrics else {}

    # Load data
    loaded_data = torch.load(join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS), weights_only=False)
    test_data = PatientDataset(loaded_data)
    test_dataset = BinaryOutcomeDataset(test_data.patients)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.folds_dir, FOLDS_FILE))
    check_for_overlap(folds, test_pids, logger)
    
    # Save predictions
    combined_df = pd.DataFrame({
        'pid': test_pids,
        'target': test_data.get_outcomes(),
    })
    if cfg.get("save_info", False):
        for k, v in cfg.save_info.items():
            func = instantiate_function(v)
            combined_df[k] = func(test_dataset)

    all_metrics = []
    for n_fold, fold in enumerate(folds, start=1):
        probas = evaluate_fold(
            finetune_folder=cfg.paths.model,
            cfg=cfg,
            test_data=test_dataset,
            logger=logger,
            fold=n_fold,
        )
        combined_df[f'fold_{n_fold}_probas'] = probas

    combined_df.to_csv(join(cfg.paths.predictions, "predictions.csv"), index=False)
    
    # # Get average metrics
    # average_metrics = {key: sum(d[key] for d in all_metrics) / len(all_metrics) for key in all_metrics[0]}
    # logger.info(f"Average metrics:")
    # for key, value in average_metrics.items():
    #     logger.info(f"{key}: {value}")

if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
