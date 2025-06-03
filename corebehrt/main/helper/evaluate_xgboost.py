from os.path import join
import pandas as pd
import xgboost as xgb
import numpy as np
import os

from corebehrt.main.helper.xgboost_cv import prepare_data_for_xgboost
from corebehrt.modules.preparation.dataset import EncodedDataset


def get_feature_importance(model: xgb.Booster, fi_cfg: dict, test_dataset: EncodedDataset, feature_names: list, logger) -> pd.DataFrame:
    """Get feature importance for a trained XGBoost model."""
    importance_type = fi_cfg.get("importance_type", "gain")
    importance = model.get_score(importance_type=importance_type)
    
    # Build dataframe with feature names
    importance_df = pd.DataFrame({
        "feature": list(importance.keys()),
        "importance": list(importance.values())
    }).sort_values("importance", ascending=False)
    
    # Add concept names by removing the "concept_" prefix
    importance_df["concept"] = importance_df["feature"].apply(
        lambda x: x.replace("concept_", "") if x.startswith("concept_") else x
    )
    
    print(importance_df)
    return importance_df

def xgb_inference_fold(model_folder: str, test_dataset: EncodedDataset, fold: int, fi_cfg: dict, logger) -> np.ndarray:
    """Run inference for a single fold using a saved Booster."""
    # Load the model
    model_path = join(model_folder, f"fold_{fold}", "xgboost_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Prepare test data
    X_test, _, feature_names, feature_types = prepare_data_for_xgboost(test_dataset, logger)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names, feature_types=feature_types)
    
    # Get predictions (probabilities)
    probas = model.predict(dtest)
    
    # Save feature importance if requested
    if fi_cfg is not None:
        logger.info(f"Getting feature importance for fold {fold}")
        fi_df = get_feature_importance(model, fi_cfg, test_dataset, feature_names, logger)
    else:
        fi_df = None
    
    return probas, fi_df