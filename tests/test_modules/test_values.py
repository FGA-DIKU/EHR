import random
import unittest
from datetime import datetime
import pandas as pd

from corebehrt.modules.features.values import ValueCreator


class TestCreators(unittest.TestCase):
    def _generate_random_pids(self):
        while True:
            yield str(random.randint(1, 4))

    def _create_concepts(self, lab_dict):
        pids = self._generate_random_pids()
        lab_df_list = []
        for concept, values in lab_dict.items():
            for value in values:
                lab_df_list.append(
                    {
                        "code": concept,
                        "numeric_value": value,
                        "time": pd.Timestamp.now(),
                        "subject_id": next(pids),
                    }
                )
        lab_df = pd.DataFrame(lab_df_list)
        return lab_df

    def setUp(self):
        self.origin_point = datetime(2020, 1, 26)
        self.background_vars = ["GENDER"]

        self.lab_dict_normed = {
            "LAB1": ["0.20", "0.30", "0.40", "0.25", "0.35", "0.21"],
            "LAB2": ["0.99", "0.81", "0.42", "0.21"],
            "LAB4": ["Kommentar"],
            "LAB5": ["0.11", "0.15", "0.12"],
            "LAB6": ["1.00"],
        }

        # Create sample data as pandas DataFrames
        self.concepts_pd_normed = self._create_concepts(self.lab_dict_normed)

    def test_create_binned_value(self):
        binned_values = ValueCreator.bin_results(self.concepts_pd_normed, num_bins=100)
        sorted_concepts = list(
            binned_values.sort_values(by=["index", "order"]).sort_index()["code"]
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


if __name__ == "__main__":
    unittest.main()
