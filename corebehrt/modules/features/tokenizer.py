import dask.dataframe as dd

from corebehrt.constants.data import (
    CLS_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    UNKNOWN_TOKEN,
)
from corebehrt.functional.features.tokenize import (
    add_special_tokens_partition,
    limit_concept_length_partition,
    tokenize_partition,
)


class EHRTokenizer:
    def __init__(
        self,
        vocabulary=None,
        cutoffs=None,
        sep_tokens: bool = True,
        cls_token: bool = True,
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
        self.sep_tokens = sep_tokens
        self.cls_token = cls_token

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
        else:
            # Ensure concepts are strings
            features["concept"] = features["concept"].astype(str)

        # Update vocabulary with concepts after cutoffs
        if self.new_vocab:
            self.update_vocabulary(features["concept"])

        # Combine all operations into a single partition pass
        def _process_partition(df):
            # Add special tokens
            if self.sep_tokens or self.cls_token:
                df = add_special_tokens_partition(
                    df, add_sep=self.sep_tokens, add_cls=self.cls_token
                )
            # Tokenize within the same partition
            df["concept"] = tokenize_partition(df["concept"], self.vocabulary)
            return df

        return features.map_partitions(_process_partition)

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
