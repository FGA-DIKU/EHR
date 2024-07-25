import dask.dataframe as dd

from corebehrt.functional.tokenize import (
    add_cls_token,
    add_separator_token,
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

    def check_cutoff(self, cutoffs: dict):
        if not isinstance(cutoffs, dict):
            raise ValueError("Cutoffs must be a dictionary")
        if not all(isinstance(value, int) for value in cutoffs.values()):
            raise ValueError("All values in cutoffs must be integers")

    def __call__(self, features: dd.DataFrame) -> dd.DataFrame:
        if self.sep_tokens:
            features = add_separator_token(features)
        if self.cls_token:
            features = add_cls_token(features)
        features = features.reset_index(drop=True)
        if self.cutoffs:
            # Cutoff concepts to max_concept_length
            features = limit_concept_length(features, self.cutoffs)

        features, vocabulary = tokenize(
            features, vocabulary=self.vocabulary, frozen_vocab=not self.new_vocab
        )
        self.vocabulary = vocabulary
        return features

    def freeze_vocabulary(self) -> None:
        self.new_vocab = False
