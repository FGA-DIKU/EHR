import logging
import os
from os.path import join

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from corebehrt.constants.paths import (
    FOLDS_FILE,
    PROCESSED_DATA_DIR,
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
from corebehrt.modules.setup.config import instantiate_class, instantiate_function
import pandas as pd

CONFIG_PATH = "./corebehrt/configs/finetune_evaluate.yaml"


def main_evaluate(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate()

    # Logger
    logger = logging.getLogger("evaluate")

    # Setup finetune config and metrics
    finetune_cfg = load_config(join(cfg.paths.model_path, FINETUNE_CFG))
    finetune_cfg.trainer_args.test_batch_size = cfg.get("test_batch_size", 128)
    metrics =(
        {k: instantiate_class(v) for k, v in cfg.metrics.items()} if cfg.metrics else {}
    )
    return_preds = True if cfg.paths.get("predictions", None) else False

    # Load data
    loaded_data = torch.load(join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS), weights_only=False)
    test_data = PatientDataset(loaded_data)
    test_dataset = BinaryOutcomeDataset(test_data.patients)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.model_path, PROCESSED_DATA_DIR, FOLDS_FILE))
    check_for_overlap(folds, test_pids, logger)
    
    if return_preds:
        combined_df = pd.DataFrame({
            'pid': test_pids,
            'target': test_data.get_outcomes(),
        })
        if cfg.get("save_info", False):
            for k, v in cfg.save_info.items():
                func = instantiate_function(v)
                combined_df[k] = func(test_dataset)

    for n_fold, fold in enumerate(folds, start=1):
        probas, metrics = evaluate_fold(
            finetune_folder=cfg.paths.model_path,
            finetune_cfg=finetune_cfg,
            test_data=test_dataset,
            logger=logger,
            fold=n_fold,
            return_preds=return_preds
        )
        combined_df[f'fold_{n_fold}_probas'] = probas

    print(combined_df.head())

if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
