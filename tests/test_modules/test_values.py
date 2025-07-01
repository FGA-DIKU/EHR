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

        self.lab_dict_normed_prefix = {
            "S/LAB1": ["0.20", "0.30", "0.40", "0.25", "0.35", "0.21"],
            "S/LAB2": ["0.99", "0.81", "0.42", "0.21"],
            "L/LAB4": ["Kommentar"],
            "L/LAB5": ["0.11", "0.15", "0.12"],
            "L/LAB6": ["1.00"],
        }

        # Create sample data as pandas DataFrames
        self.concepts_pd_normed = self._create_concepts(self.lab_dict_normed)
        self.concepts_pd_normed_prefix = self._create_concepts(
            self.lab_dict_normed_prefix
        )

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

    def test_create_binned_value_with_prefix(self):
        binned_values = ValueCreator.bin_results(
            self.concepts_pd_normed_prefix,
            num_bins=100,
            add_prefix=True,
            prefix_regex=r"^([^/]+)/",
        )
        sorted_concepts = list(
            binned_values.sort_values(by=["index", "order"]).sort_index()["code"]
        )

        expected_binned_concepts = []
        for lab, values in self.lab_dict_normed_prefix.items():
            prefix = lab.split("/")[0]  # Extract prefix from key
            for value in values:
                expected_binned_concepts.append(
                    [
                        lab,
                        f"{prefix}/VAL_{int(float(value) * 100)}"
                        if value != "Kommentar"
                        else None,
                    ]
                )
        expected_flattened_binned_concepts = [
            item
            for sublist in expected_binned_concepts
            for item in sublist
            if item is not None
        ]
        self.assertEqual(sorted_concepts, expected_flattened_binned_concepts)

    def test_all_nan_values(self):
        """Test that the bin_results method handles cases where all numeric values are NaN."""
        # Create a DataFrame with only NaN values in numeric_value
        nan_df = pd.DataFrame(
            {
                "code": ["LAB_NAN1", "LAB_NAN2", "LAB_NAN3"],
                "numeric_value": [float("nan"), float("nan"), float("nan")],
                "time": [pd.Timestamp.now()] * 3,
                "subject_id": ["1", "2", "3"],
            }
        )

        # This should not raise an error
        try:
            binned_values = ValueCreator.bin_results(nan_df, num_bins=100)
            self.assertIsInstance(binned_values, pd.DataFrame)
        except ValueError as e:
            self.fail(f"bin_results raised ValueError with all-NaN input: {e}")

    def test_empty_dataframe(self):
        """Test that the bin_results method handles empty DataFrames correctly."""
        # Create an empty DataFrame with only headers
        empty_df = pd.DataFrame(columns=["code", "numeric_value", "time", "subject_id"])

        # This should not raise an error
        try:
            binned_values = ValueCreator.bin_results(empty_df, num_bins=100)
            self.assertIsInstance(binned_values, pd.DataFrame)
        except ValueError as e:
            self.fail(f"bin_results raised ValueError with empty DataFrame: {e}")

    def test_single_row_nan(self):
        """Test that the bin_results method handles a single row with NaN."""
        # Create a DataFrame with a single row containing NaN
        single_nan_df = pd.DataFrame(
            {
                "code": ["LAB_SINGLE"],
                "numeric_value": [float("nan")],
                "time": [pd.Timestamp.now()],
                "subject_id": ["1"],
            }
        )

        # This should not raise an error
        try:
            binned_values = ValueCreator.bin_results(single_nan_df, num_bins=100)
            self.assertIsInstance(binned_values, pd.DataFrame)
        except ValueError as e:
            self.fail(f"bin_results raised ValueError with single NaN value: {e}")


if __name__ == "__main__":
    unittest.main()
