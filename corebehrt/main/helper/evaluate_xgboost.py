from os.path import join
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import torch

from corebehrt.main.helper.xgboost_cv import prepare_data_for_xgboost
from corebehrt.modules.preparation.dataset import EncodedDataset


def get_feature_importance(model: xgb.Booster, fi_cfg: dict, feature_names: list) -> pd.DataFrame:
    """Get feature importance for a trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        fi_cfg: Feature importance configuration
        feature_names: List of feature names
    """
    importance_type = fi_cfg.get("importance_type", "gain")
    importance = model.get_score(importance_type=importance_type)
    
    # Create a dictionary with all features, defaulting to 0 importance
    all_importance = {name: 0.0 for name in feature_names}
    # Update with actual importance scores
    all_importance.update(importance)
    
    # Build dataframe with feature names and extract concepts
    importance_df = pd.DataFrame({
        "concept": [x.split("_", 1)[1] if x.startswith("concept_") else x for x in all_importance.keys()],
        "importance": list(all_importance.values())
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    print(f"\nTotal number of features: {len(importance_df)}")
    print(f"Number of features with non-zero importance: {len(importance)}")
    
    return importance_df

def xgb_inference_fold(model_folder: str, test_dataset: EncodedDataset, fold: int, fi_cfg: dict, logger, vocab_mapping_path: str = None) -> np.ndarray:
    """Run inference for a single fold using a saved Booster."""
    # Load the model
    model_path = join(model_folder, f"fold_{fold}", "xgboost_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Prepare test data
    X_test, y_test, feature_names, feature_types = prepare_data_for_xgboost(test_dataset, logger)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names, feature_types=feature_types)
    
    # Get predictions (probabilities)
    probas = model.predict(dtest)
    
    # Save feature importance if requested
    if fi_cfg is not None:
        logger.info(f"Getting feature importance for fold {fold}")
        fi_df = get_feature_importance(model, fi_cfg, feature_names)
    else:
        fi_df = None
    
    return probas, fi_df