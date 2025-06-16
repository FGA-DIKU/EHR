from os.path import join
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import torch
from typing import Tuple, Optional

from corebehrt.modules.preparation.encode import OneHotEncoder


def get_feature_importance(
    model: xgb.Booster, fi_cfg: dict, encoding_vocab: dict
) -> pd.DataFrame:
    """Get feature importance for a trained XGBoost model."""
    importance_type = fi_cfg.get("importance_type", "gain")
    importance = model.get_score(importance_type=importance_type)
    rev_encoding_vocab = {v: k for k, v in encoding_vocab.items()}
    importance_df = pd.DataFrame(
        {
            "feature": list(importance.keys()),
            "concept": [
                rev_encoding_vocab[int(feature.lstrip("f"))]
                for feature in importance.keys()
            ],
            "importance": list(importance.values()),
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    return importance_df


def xgb_inference_fold(
    model_folder: str, test_data, fold: int, fi_cfg: dict, logger, vocab: dict
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """Run inference for a single fold using a saved Booster and the saved encoding mapping."""
    # Load the model
    model_path = join(model_folder, f"fold_{fold}", "xgboost_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)

    # Load the encoding mapping used during training
    encoding_vocab = torch.load(join(model_folder, f"fold_{fold}", "encoding_vocab.pt"))
    encoder = OneHotEncoder(vocabulary=vocab, encoding_vocab=encoding_vocab)
    (
        X_test,
        _,
    ) = encoder.to_xgboost(test_data.patients)
    dtest = xgb.DMatrix(X_test)
    print(
        [
            (concept, val)
            for concept, val in encoding_vocab.items()
            if concept.startswith("S")
        ]
    )

    # Get predictions (probabilities)
    probas = model.predict(dtest)

    # Save feature importance if requested
    if fi_cfg is not None:
        logger.info(f"Getting feature importance for fold {fold}")
        fi_df = get_feature_importance(model, fi_cfg, encoding_vocab)
    else:
        fi_df = None

    return probas, fi_df
