import os
from os.path import join
from typing import List

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from corebehrt.azure import log_metrics, setup_metrics_dir
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.modules.preparation.dataset import PatientDataset, EncodedDataset
import xgboost as xgb
import logging
from corebehrt.modules.setup.config import instantiate_function
from corebehrt.main.helper.finetune_cv import log_best_metrics

def prepare_data_for_xgboost(dataset, logger):
    """Convert encoded dataset to XGBoost format."""
    all_data = [dataset[i] for i in range(len(dataset))]
    
    # Get the one-hot encoded concepts, age at censoring, and outcomes
    X_concepts = np.array([d['concepts'] for d in all_data])
    X_age = np.array([d['age_at_censoring'] for d in all_data]).reshape(-1, 1)
    X = np.hstack([X_concepts, X_age])  # Combine concepts and age features
    y = np.array([d['outcome'] for d in all_data])
    
    if logger:
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Unique outcomes: {np.unique(y, return_counts=True)}")
        logger.info(f"Age range: [{X_age.min()}, {X_age.max()}]")
        logger.info(f"Non-zero features: {np.count_nonzero(X)}")
    
    return X, y


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
    val_dataset = EncodedDataset(val_data.patients, vocab)
    test_dataset = EncodedDataset(test_data.patients, vocab) if test_data and len(test_data) > 0 else None

    X_train, y_train = prepare_data_for_xgboost(train_dataset, logger)
    X_val, y_val = prepare_data_for_xgboost(val_dataset, logger)
    if test_dataset:
        X_test, y_test = prepare_data_for_xgboost(test_dataset, logger)

    # Prepare DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    if test_dataset:
        dtest = xgb.DMatrix(X_test, label=y_test)

    # Parameters for xgb.train
    params = dict(cfg.model)  # copy config

    logger.info(f"Starting XGBoost training for fold {fold}...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, 'train'), (dval, 'val')],
        verbose_eval=True,
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