import numpy as np
import logging

from corebehrt.constants.data import (
    PAD_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
    UNKNOWN_TOKEN,
    MASK_TOKEN,
    AGE_AT_CENSORING_TOKEN,
)

logger = logging.getLogger(__name__)  # Get the logger for this module


class OneHotEncoder:
    def __init__(self, vocabulary: dict, encoding_vocab: dict = None):
        if vocabulary is None:
            raise ValueError("Provide vocabulary.")

        special_tokens = [
            PAD_TOKEN,
            CLS_TOKEN,
            SEP_TOKEN,
            UNKNOWN_TOKEN,
            MASK_TOKEN,
            AGE_AT_CENSORING_TOKEN,
        ]
        self.vocabulary = vocabulary

        if encoding_vocab is None:
            new_idx = 0
            encoding_vocab = {}
            original_to_new_idx = {}
            for old_idx, token in enumerate(vocabulary.keys()):
                if token not in special_tokens:
                    encoding_vocab[token] = new_idx
                    original_to_new_idx[old_idx] = new_idx
                    new_idx += 1
            encoding_vocab[AGE_AT_CENSORING_TOKEN] = new_idx
            self.encoding_vocab = encoding_vocab
            self.original_to_new_idx = original_to_new_idx
        else:
            self.encoding_vocab = encoding_vocab
            self.original_to_new_idx = {
                vocabulary[token]: idx
                for token, idx in encoding_vocab.items()
                if token not in special_tokens
            }

        self.num_features = len(self.encoding_vocab)

    def to_xgboost(self, patients):
        """Convert patients directly to XGBoost format (X, y)."""
        # Create a binary vector for each patient's concepts
        X_concepts = np.zeros((len(patients), self.num_features - 1), dtype=np.int16)
        age_at_censoring = []
        y = []
        for i, patient in enumerate(patients):
            unique_concepts = np.unique(patient.concepts)
            for concept in unique_concepts:
                if concept in self.original_to_new_idx:
                    new_idx = self.original_to_new_idx[concept]
                    X_concepts[i, new_idx] = 1
            age_at_censoring.append(patient.ages[-1] if patient.ages else 0)
            y.append(patient.outcome if patient.outcome is not None else None)
        X_age = np.array(age_at_censoring).reshape(-1, 1)
        X = np.hstack([X_concepts, X_age])
        y = np.array(y)
        return X, y
