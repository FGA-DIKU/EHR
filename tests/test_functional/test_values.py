import random
import unittest
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd

from corebehrt.modules.features.normalizer import ValuesNormalizer
from corebehrt.modules.features.values import ValueCreator


class TestCreators(unittest.TestCase):
    def _generate_random_pids(self):
        while True:
            yield str(random.randint(1, 4))

    def _create_concepts(self, lab_dict):
        admission_ids = (chr(i) for i in iter(range(65, 91)))
        pids = self._generate_random_pids()
        lab_df_list = []
        for concept, values in lab_dict.items():
            for value in values:
                lab_df_list.append(
                    {
                        "CONCEPT": concept,
                        "RESULT": value,
                        "TIMESTAMP": pd.Timestamp.now(),
                        "ADMISSION_ID": next(admission_ids),
                        "PID": next(pids),
                    }
                )
        lab_df = pd.DataFrame(lab_df_list)
        return lab_df

    def _min_max_norm(self, x, values):
        if len(values) < 3:
            return -1
        f_x = float(x)
        f_values = np.array([float(value) for value in values])
        return (f_x - f_values.min()) / (f_values.max() - f_values.min())

    def setUp(self):
        self.origin_point = datetime(2020, 1, 26)
        self.background_vars = ["GENDER"]

        self.lab_dict = {
            "LAB1": ["20", "30", "40", "25", "35", "20"],
            "LAB2": ["1.1", "0.8", "0.4", "0.2"],
            "LAB4": ["Kommentar"],
            "LAB5": ["100", "150", "100"],
            "LAB6": ["1000"],
        }

        self.lab_dict_normed = {
            "LAB1": ["0.20", "0.30", "0.40", "0.25", "0.35", "0.21"],
            "LAB2": ["0.99", "0.81", "0.42", "0.21"],
            "LAB4": ["Kommentar"],
            "LAB5": ["0.11", "0.15", "0.12"],
            "LAB6": ["1.00"],
        }

        self.lab_dict_quantiles = {
            "LAB1": ["Q1", "Q2", "Q3", "Q1", "Q2", "Q1"],
            "LAB2": ["Q1", "Q1", "Q1", "Q1"],
            "LAB4": ["Kommentar"],
            "LAB5": ["Q1", "Q2", "Q1"],
            "LAB6": ["Q1"],
        }

        # Create sample data as pandas DataFrames
        self.concepts_pd = self._create_concepts(self.lab_dict)
        self.concepts_pd_normed = self._create_concepts(self.lab_dict_normed)
        self.concepts_pd_quantiles = self._create_concepts(self.lab_dict_quantiles)

        self.patients_info_pd = pd.DataFrame(
            {
                "PID": ["1", "2", "3", "4"],
                "BIRTHDATE": pd.to_datetime(
                    ["2000-01-02", "2000-03-20", "2000-05-08", "2000-07-01"]
                ),
                "GENDER": ["Male", "Female", "Male", "Female"],
            }
        )

        # Convert pandas DataFrames to Dask DataFrames
        self.concepts = dd.from_pandas(self.concepts_pd, npartitions=1)
        self.concepts_normed = dd.from_pandas(self.concepts_pd_normed, npartitions=1)
        self.concepts_quantiles = dd.from_pandas(
            self.concepts_pd_quantiles, npartitions=1
        )
        self.patients_info = dd.from_pandas(self.patients_info_pd, npartitions=1)

    def test_create_binned_value(self):
        binned_values = ValueCreator.add_binned_values(
            self.concepts_normed, multiplication_factor=100
        ).compute()
        sorted_concepts = list(
            binned_values.sort_values(by=["index", "order"]).sort_index()["CONCEPT"]
        )
        expected_binned_concepts = [
            [
                lab,
                "VAL_" + str(int(float(value) * 100)) if value != "Kommentar" else None,
            ]
            for lab, values in self.lab_dict_normed.items()
            for value in values
        ]
        expected_flattened_binned_concepts = [
            item
            for sublist in expected_binned_concepts
            for item in sublist
            if item is not None
        ]
        self.assertEqual(sorted_concepts, expected_flattened_binned_concepts)

    def test_normalise_and_create_binned_value(self):
        binned_values = ValueCreator.add_binned_values(
            self.concepts,
            multiplication_factor=100,
            normalize_args={
                "func": ValuesNormalizer.min_max_normalize_results,
                "kwargs": {"min_count": 3},
            },
        ).compute()
        sorted_concepts = list(
            binned_values.sort_values(by=["index", "order"]).sort_index()["CONCEPT"]
        )
        expected_binned_concepts = [
            [
                lab,
                (
                    "VAL_" + str(int(self._min_max_norm(value, values) * 100))
                    if value != "Kommentar"
                    else None
                ),
            ]
            for lab, values in self.lab_dict.items()
            for value in values
        ]
        expected_flattened_binned_concepts = [
            item
            for sublist in expected_binned_concepts
            for item in sublist
            if item is not None
        ]

        self.assertEqual(sorted_concepts, expected_flattened_binned_concepts)

    def test_create_quantile_value(self):
        quantile_values = ValueCreator.add_quantile_values(
            self.concepts_quantiles
        ).compute()
        sorted_concepts = list(
            quantile_values.sort_values(by=["index", "order"]).sort_index()["CONCEPT"]
        )
        expected_quantile_concepts = [
            [lab, "VAL_" + value if value.startswith("Q") else None]
            for lab, values in self.lab_dict_quantiles.items()
            for value in values
        ]
        expected_flattened_quantile_concepts = [
            item
            for sublist in expected_quantile_concepts
            for item in sublist
            if item is not None
        ]

        self.assertEqual(sorted_concepts, expected_flattened_quantile_concepts)


if __name__ == "__main__":
    unittest.main()
