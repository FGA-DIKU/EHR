import logging
from os.path import join
import pandas as pd
import torch
import os
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from corebehrt.constants.paths import FOLDS_FILE, PREPARED_ALL_PATIENTS
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.xgboost_cv import prepare_data_for_xgboost
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main.helper.evaluate_finetune import compute_metrics
from corebehrt.modules.preparation.dataset import EncodedDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.functional.io_operations.load import load_vocabulary

from corebehrt.constants.paths import XGBOOST_CFG

CONFIG_PATH = "./corebehrt/configs/evaluate_xgboost.yaml"


def inference_fold(model_folder: str, test_dataset: EncodedDataset, fold: int, logger) -> np.ndarray:
    """Run inference for a single fold using a saved Booster."""
    # Load the model
    model_path = join(model_folder, f"fold_{fold}", "xgboost_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Prepare test data
    X_test, _ = prepare_data_for_xgboost(test_dataset, logger)
    dtest = xgb.DMatrix(X_test)
    
    # Get predictions (probabilities)
    probas = model.predict(dtest)
    
    return probas


def main_evaluate(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate(finetune_cfg=XGBOOST_CFG)

    # Logger
    logger = logging.getLogger("evaluate")

    # Load data
    loaded_data = torch.load(join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS))
    test_data = PatientDataset(loaded_data)
    vocab = load_vocabulary(cfg.paths.test_data_dir)
    test_dataset = EncodedDataset(test_data.patients, vocab)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.folds_dir, FOLDS_FILE))
    check_for_overlap(folds, test_pids, logger)
    targets = test_data.get_outcomes()

    # Get predictions
    combined_df = pd.DataFrame({
        "pid": test_pids,
        "target": targets,
    })

    all_probas = []
    for n_fold, fold in enumerate(folds, start=1):
        logger.info(f"Evaluating fold {n_fold}/{len(folds)}")
        probas = inference_fold(
            model_folder=cfg.paths.model,
            test_dataset=test_dataset,
            fold=n_fold,
            logger=logger
        )
        combined_df[f"fold_{n_fold}_probas"] = probas
        all_probas.append(probas)

    # Save predictions
    combined_df.to_csv(join(cfg.paths.predictions, "predictions.csv"), index=False)
    
    # Compute and log metrics
    compute_metrics(cfg, targets, all_probas, logger)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
