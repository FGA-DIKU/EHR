import dask.dataframe as dd

from corebehrt.functional.constants import (
    CLS_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    UNKNOWN_TOKEN,
)
from corebehrt.functional.tokenize import (
    limit_concept_length_partition,
    tokenize_partition,
)


class EHRTokenizer:
    def __init__(
        self,
        vocabulary=None,
        cutoffs=None,
    ):
        if vocabulary is None:
            self.new_vocab = True
            self.vocabulary = {
                PAD_TOKEN: 0,
                CLS_TOKEN: 1,
                SEP_TOKEN: 2,
                UNKNOWN_TOKEN: 3,
                MASK_TOKEN: 4,
            }
        else:
            self.new_vocab = False
            self.vocabulary = vocabulary

        if cutoffs is not None:
            self.check_cutoff(cutoffs)
        self.cutoffs = cutoffs

    def check_cutoff(self, cutoffs) -> None:
        if not isinstance(cutoffs, dict):
            raise ValueError("Cutoffs must be a dictionary")
        if not all(isinstance(value, int) for value in cutoffs.values()):
            raise ValueError("All values in cutoffs must be integers")

    def __call__(self, features: dd.DataFrame) -> dd.DataFrame:
        """
        !We assume that features are sorted by PID and abspos and PID is the index.
        """
        # Apply cutoffs if needed before updating vocabulary
        if self.cutoffs:
            features["concept"] = features["concept"].map_partitions(
                limit_concept_length_partition, self.cutoffs
            )

        # Update vocabulary with concepts after cutoffs
        if self.new_vocab:
            self.update_vocabulary(features["concept"])

        # Tokenize within the same partition
        def _tokenize_partition(series):
            return tokenize_partition(series, self.vocabulary)

        features["concept"] = features["concept"].map_partitions(_tokenize_partition)
        return features

    def update_vocabulary(self, concepts: dd.Series) -> None:
        """Create or update vocabulary from unique concepts"""
        # Get unique concepts across all partitions
        unique_concepts = concepts.drop_duplicates().compute()

        # Start with base vocabulary
        vocabulary = self.vocabulary.copy()
        # Add new concepts
        new_concepts = set(unique_concepts) - set(vocabulary)
        # Pre-calculate the starting index
        start_idx = max(vocabulary.values()) + 1
        if new_concepts:
            new_indices = range(start_idx, start_idx + len(new_concepts))
            vocabulary.update(dict(zip(new_concepts, new_indices)))
        self.vocabulary = vocabulary

    def freeze_vocabulary(self) -> None:
        self.new_vocab = False
