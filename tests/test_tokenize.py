import unittest
import dask.dataframe as dd
import pandas as pd

from corebehrt.functional.tokenize import (
    add_separator_token,
    add_cls_token,
    limit_concept_length,
    tokenize,
)


class TestCreators(unittest.TestCase):
    def setUp(self):
        self.features = dd.from_pandas(
            pd.DataFrame(
                {
                    "PID": map(str, [1, 1, 1, 2, 2, 3, 3, 3, 3]),
                    "concept": ["C1", "C2", "C2", "C2", "C3", "C4a", "C2", "C2", "C4b"],
                    "age": [33.1, 33.2, 33.3, 21.9, 22.0, 36.1, 36.7, 38.1, 38.2],
                    "segment": [0, 1, 1, 0, 1, 0, 1, 2, 2],
                    "abspos": map(float, [1, 2, 3, 1, 2, 1, 2, 3, 4]),
                }
            ).set_index("PID"),
        ).reset_index()

        self.vocabulary = {"[UNK]": 0, "C1": 1, "C2": 2, "C3": 3}

    def test_add_sep(self):
        expected_concept = [
            "C1",
            "[SEP]",
            "C2",
            "C2",
            "[SEP]",
            "C2",
            "[SEP]",
            "C3",
            "[SEP]",
            "C4a",
            "[SEP]",
            "C2",
            "[SEP]",
            "C2",
            "C4b",
            "[SEP]",
        ]
        expected_segment = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2]

        result = add_separator_token(self.features)

        self.assertIsNot(result, self.features)
        result = result.compute()
        self.assertEqual(result.concept.tolist(), expected_concept)
        self.assertEqual(result.segment.tolist(), expected_segment)

    def test_add_cls(self):
        expected_concept = [
            "[CLS]",
            "C1",
            "C2",
            "C2",
            "[CLS]",
            "C2",
            "C3",
            "[CLS]",
            "C4a",
            "C2",
            "C2",
            "C4b",
        ]
        expected_segment = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 2, 2]

        result = add_cls_token(self.features)

        self.assertIsNot(result, self.features)
        result = result.compute()
        self.assertEqual(result.concept.tolist(), expected_concept)
        self.assertEqual(result.segment.tolist(), expected_segment)

    def test_limit_concept_length(self):
        cutoffs = {"C4": 2}
        expected_concept = ["C1", "C2", "C2", "C2", "C3", "C4", "C2", "C2", "C4"]

        result_concept = limit_concept_length(self.features.concept, cutoffs)
        result_concept = result_concept.compute().tolist()

        self.assertEqual(result_concept, expected_concept)

    def test_tokenize_update(self):
        expected_concept = [1, 2, 2, 2, 3, 4, 2, 2, 5]
        expected_vocabulary = {
            "[UNK]": 0,
            "C1": 1,
            "C2": 2,
            "C3": 3,
            "C4a": 4,
            "C4b": 5,
        }

        result_concept, result_vocab = tokenize(
            self.features.concept, self.vocabulary, False
        )
        result_concept = result_concept.compute().tolist()

        self.assertEqual(result_concept, expected_concept)
        self.assertEqual(result_vocab, expected_vocabulary)

    def test_tokenize_frozen(self):
        expected_concept = [1, 2, 2, 2, 3, 0, 2, 2, 0]

        result_concept, result_vocab = tokenize(
            self.features.concept, self.vocabulary, True
        )
        result_concept = result_concept.compute().tolist()

        self.assertIs(result_vocab, self.vocabulary)
        self.assertEqual(result_concept, expected_concept)
