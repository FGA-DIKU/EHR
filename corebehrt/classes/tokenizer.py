import dask.dataframe as dd

from corebehrt.functional.tokenize import (
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
                "[PAD]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[UNK]": 3,
                "[MASK]": 4,
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
        # Avoid index operations altogether
        meta = features._meta.copy()
        if self.new_vocab:
            self.update_vocabulary(features["concept"])

        # Combine all operations into a single partition pass
        def _process_partition(df):
            # Add special tokens
            if self.sep_tokens or self.cls_token:
                df = add_special_tokens_partition(
                    df, add_sep=self.sep_tokens, add_cls=self.cls_token
                )
            # Apply cutoffs if needed
            if self.cutoffs:
                df["concept"] = limit_concept_length_partition(
                    df["concept"], self.cutoffs
                )

            # Tokenize within the same partition
            df["concept"] = tokenize_partition(df["concept"], self.vocabulary)
            return df

        return features.map_partitions(_process_partition, meta=meta)

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
