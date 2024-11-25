import dask.dataframe as dd

from corebehrt.functional.tokenize import (
    add_special_tokens_partition,
    limit_concept_length,
    tokenize,
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

    def check_cutoff(self, cutoffs):
        if not isinstance(cutoffs, dict):
            raise ValueError("Cutoffs must be a dictionary")
        if not all(isinstance(value, int) for value in cutoffs.values()):
            raise ValueError("All values in cutoffs must be integers")

    def __call__(self, features: dd.DataFrame) -> dd.DataFrame:
        features = features.set_index("PID", sorted=False, drop=False)
        # Apply special tokens if needed
        if self.sep_tokens or self.cls_token:
            features = features.map_partitions(
                add_special_tokens_partition,
                add_sep=self.sep_tokens,
                add_cls=self.cls_token,
                sep_token="[SEP]",
                cls_token="[CLS]",
            )
            features = features.set_index("PID", sorted=False, drop=False)

        # Apply cutoffs if needed
        if self.cutoffs:
            features["concept"] = limit_concept_length(
                features["concept"], self.cutoffs
            )

        # Tokenize
        features["concept"], vocabulary = tokenize(
            features["concept"],
            vocabulary=self.vocabulary,
            frozen_vocab=not self.new_vocab,
        )
        self.vocabulary = vocabulary

        return features

    def freeze_vocabulary(self) -> None:
        self.new_vocab = False
