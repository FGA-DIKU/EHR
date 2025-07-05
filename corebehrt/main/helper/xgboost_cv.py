import os
from os.path import join
import torch
import xgboost as xgb

from corebehrt.azure.util.log import setup_metrics_dir
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.setup.config import instantiate_function
from corebehrt.modules.preparation.encode import OneHotEncoder


def log_metrics_dict(metrics: dict, split: str) -> None:
    """
    Logs a dict of metrics with proper prefixing for the given split.
    """
    if metrics:
        try:
            import mlflow
            if mlflow.active_run() is not None:
                # Apply split prefix to all metric keys
                prefixed_metrics = {f"{split}_{k}": v for k, v in metrics.items()}
                mlflow.log_metrics(prefixed_metrics)
        except ImportError:
            # mlflow not available, skip logging
            pass


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
    encoder = OneHotEncoder(vocabulary=vocab)
    train_patients = train_data.patients
    val_patients = val_data.patients
    test_patients = test_data.patients if test_data and len(test_data) > 0 else None
    encoding_vocab = encoder.encoding_vocab
    torch.save(encoding_vocab, join(fold_folder, "encoding_vocab.pt"))

    (
        X_train,
        y_train,
    ) = encoder.to_xgboost(train_patients)
    X_val, y_val = encoder.to_xgboost(val_patients)
    if test_patients:
        X_test, y_test = encoder.to_xgboost(test_patients)

    # Prepare DMatrix without feature names/types
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    if test_patients:
        dtest = xgb.DMatrix(X_test, label=y_test)

    # Parameters for xgb.train
    params = dict(cfg.model)  # copy config

    logger.info(f"Starting XGBoost training for fold {fold}...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        **cfg.trainer_args,
    )

    # Predict
    train_preds = model.predict(dtrain)
    val_preds = model.predict(dval)
    if test_patients:
        test_preds = model.predict(dtest)

    # Compute metrics using config-defined metrics
    if hasattr(cfg, "metrics") and cfg.metrics:
        metrics = {k: instantiate_function(v) for k, v in cfg.metrics.items()}

        # Compute train metrics
        train_metrics = {
            f"train_{name}": func(y_train, train_preds)
            for name, func in metrics.items()
        }
        log_metrics_dict(train_metrics, "train")
        logger.info(f"Train metrics: {train_metrics}")

        # Compute validation metrics
        val_metrics = {
            f"val_{name}": func(y_val, val_preds) for name, func in metrics.items()
        }
        log_metrics_dict(val_metrics, "val")
        logger.info(f"Val metrics: {val_metrics}")

        # Compute test metrics if available
        if test_patients is not None:
            test_metrics = {
                f"test_{name}": func(y_test, test_preds)
                for name, func in metrics.items()
            }
            log_metrics_dict(test_metrics, "test")
            logger.info(f"Test metrics: {test_metrics}")

    # Save model
    model.save_model(join(fold_folder, "xgboost_model.json"))
    logger.info(f"Model saved to {join(fold_folder, 'xgboost_model.json')}")
