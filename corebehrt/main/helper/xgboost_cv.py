import os
from os.path import join
from typing import List

import torch
import numpy as np

from corebehrt.azure import setup_metrics_dir
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.modules.preparation.dataset import PatientDataset, EncodedDataset
import xgboost as xgb
from corebehrt.modules.setup.config import instantiate_function
from corebehrt.main.helper.finetune_cv import log_best_metrics

def prepare_data_for_xgboost(dataset, logger):
    """Convert encoded dataset to XGBoost format."""

    concepts = [dataset[i]['concepts'] for i in range(len(dataset))]
    age_at_censoring = [dataset[i]['age_at_censoring'] for i in range(len(dataset))]
    X_concepts = np.array(concepts)
    X_age = np.array(age_at_censoring).reshape(-1, 1)
    X = np.hstack([X_concepts, X_age])
    
    # Create feature names using vocabulary
    feature_names = []
    # Create reverse mapping from index to code
    idx_to_code = dataset.idx_to_token
    
    # Use only valid indices in the same order as the one-hot encoding
    for orig_idx in dataset.valid_indices:
        code = idx_to_code[orig_idx]
        feature_names.append(f"concept_{code}")
    feature_names.append("age_at_censoring")
    
    # Create feature types (all categorical)
    feature_types = ['c'] * len(dataset.valid_indices) + ['q']
    
    y = np.array([dataset[i]['outcome'] for i in range(len(dataset))])
    
    return X, y, feature_names, feature_types


def cv_loop(
    cfg,
    logger,
    model_folder: str,
    data: PatientDataset,
    vocab: dict,
    folds: list,
    test_data: PatientDataset,
) -> None:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    for fold, fold_dict in enumerate(folds):
        fold += 1  # 1-indexed
        train_pids = fold_dict[TRAIN_KEY]
        val_pids = fold_dict[VAL_KEY]
        logger.info(f"Training fold {fold}/{len(folds)}")

        train_data = data.filter_by_pids(train_pids)
        val_data = data.filter_by_pids(val_pids)

        with setup_metrics_dir(f"Fold {fold}"):
            xgboost_fold(
                cfg, logger, model_folder, train_data, val_data, fold, test_data, vocab
            )

def xgboost_fold(
    cfg,
    logger,
    model_folder: str,
    train_data: PatientDataset,
    val_data: PatientDataset,
    fold: int,
    test_data: PatientDataset = None,
    vocab: dict = None,
) -> None:
    """Train XGBoost model on one fold using Booster."""

    fold_folder = join(model_folder, f"fold_{fold}")
    os.makedirs(fold_folder, exist_ok=True)

    logger.info("Saving pids")
    torch.save(train_data.get_pids(), join(fold_folder, "train_pids.pt"))
    torch.save(val_data.get_pids(), join(fold_folder, "val_pids.pt"))
    if test_data and len(test_data) > 0:
        torch.save(test_data.get_pids(), join(fold_folder, "test_pids.pt"))

    logger.info("Initializing datasets")
    train_dataset = EncodedDataset(train_data.patients, vocab)

    print(train_data.patients[0].concepts)
        
    val_dataset = EncodedDataset(val_data.patients, vocab)
    test_dataset = EncodedDataset(test_data.patients, vocab) if test_data and len(test_data) > 0 else None

    X_train, y_train, feature_names_train, feature_types_train = prepare_data_for_xgboost(train_dataset, logger)
    X_val, y_val, feature_names_val, feature_types_val = prepare_data_for_xgboost(val_dataset, logger)
    if test_dataset:
        X_test, y_test, feature_names_test, feature_types_test = prepare_data_for_xgboost(test_dataset, logger)

    # Prepare DMatrix with feature types
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names_train, feature_types=feature_types_train)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names_val, feature_types=feature_types_val)
    if test_dataset:
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names_test, feature_types=feature_types_test)

    # Parameters for xgb.train
    params = dict(cfg.model)  # copy config

    logger.info(f"Starting XGBoost training for fold {fold}...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, 'train'), (dval, 'val')],
        **cfg.trainer_args
    )

    # Predict
    train_preds = model.predict(dtrain)
    val_preds = model.predict(dval)
    if test_dataset:
        test_preds = model.predict(dtest)

    # Compute metrics using config-defined metrics
    if hasattr(cfg, "metrics") and cfg.metrics:
        metrics = {k: instantiate_function(v) for k, v in cfg.metrics.items()}
        
        # Compute train metrics
        train_metrics = {f"train_{name}": func(y_train, train_preds) for name, func in metrics.items()}
        log_best_metrics(0.0, train_metrics, "train")
        logger.info(f"Train metrics: {train_metrics}")
        
        # Compute validation metrics
        val_metrics = {f"val_{name}": func(y_val, val_preds) for name, func in metrics.items()}
        log_best_metrics(0.0, val_metrics, "val")
        logger.info(f"Val metrics: {val_metrics}")
        
        # Compute test metrics if available
        if test_dataset:
            test_metrics = {f"test_{name}": func(y_test, test_preds) for name, func in metrics.items()}
            log_best_metrics(0.0, test_metrics, "test")
            logger.info(f"Test metrics: {test_metrics}")
            
    # Save model
    model.save_model(join(fold_folder, 'xgboost_model.json'))
    logger.info(f"Model saved to {join(fold_folder, 'xgboost_model.json')}")