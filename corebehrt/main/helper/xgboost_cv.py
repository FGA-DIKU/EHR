import os
from os.path import join
from typing import List

import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

from corebehrt.azure import log_metrics, setup_metrics_dir
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.constants.train import DEFAULT_VAL_SPLIT
from corebehrt.functional.features.split import get_n_splits_cv_pids
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.modules.preparation.dataset import PatientDataset, EncodedDataset
from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.trainer import EHRTrainer
import logging


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
    """Train XGBoost model on one fold"""

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

    # Prepare data for XGBoost
    logger.info("Preparing training data...")
    X_train, y_train = prepare_data_for_xgboost(train_dataset, logger)
    logger.info("Preparing validation data...")
    X_val, y_val = prepare_data_for_xgboost(val_dataset, logger)
    if test_dataset:
        logger.info("Preparing test data...")
        X_test, y_test = prepare_data_for_xgboost(test_dataset, logger)

    # Initialize XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'random_state': 42
    }

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    if test_dataset:
        dtest = xgb.DMatrix(X_test, label=y_test)

    # Train the model
    logger.info(f"Starting XGBoost training for fold {fold}...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=True
    )

    # Make predictions
    train_preds = model.predict(dtrain)
    val_preds = model.predict(dval)
    if test_dataset:
        test_preds = model.predict(dtest)

    # Calculate metrics
    train_auc = roc_auc_score(y_train, train_preds)
    val_auc = roc_auc_score(y_val, val_preds)
    train_acc = accuracy_score(y_train, train_preds > 0.5)
    val_acc = accuracy_score(y_val, val_preds > 0.5)

    # Log metrics
    metrics = {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    log_best_metrics(0.0, metrics, 'val')  # Using 0.0 as loss since XGBoost doesn't use it

    if test_dataset:
        test_auc = roc_auc_score(y_test, test_preds)
        test_acc = accuracy_score(y_test, test_preds > 0.5)
        test_metrics = {
            'test_auc': test_auc,
            'test_acc': test_acc
        }
        log_best_metrics(0.0, test_metrics, 'test')

    # Save the model
    model.save_model(join(fold_folder, 'xgboost_model.json'))
    logger.info(f"Model saved to {join(fold_folder, 'xgboost_model.json')}")


def log_best_metrics(loss: float, metrics: dict, split: str) -> None:
    """
    Logs a dict of metrics, where each metric is prepended by 'best.<split>.'.
    Example: 'val_loss' -> 'best.val.val_loss'
    """
    row = {f"{split}_loss": loss, **metrics}
    prefixed = {f"best.{split}.{k}": v for k, v in row.items()}
    log_metrics(prefixed)


def check_for_overlap(folds: List[dict], test_pids: list, logger) -> None:
    """
    Check for overlap between test and train/validation patient IDs.
    """
    fold = folds[0]  # all folds have same pids in total, we use fold 0 as example

    train_pids = set(fold[TRAIN_KEY])
    val_pids = set(fold[VAL_KEY])
    test_pids = set(test_pids)
    if train_pids & test_pids or val_pids & test_pids:
        logger.warning(
            "Found overlap between test and train/validation patient IDs. "
            "This means some patients appear in both test and training/validation sets, "
            "which may lead to data leakage and overly optimistic results. "
            "Please verify this overlap is intentional for your use case."
        )
