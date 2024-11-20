import unittest
import pandas as pd
import dask.dataframe as dd
from corebehrt.functional.utils import (
    truncate_patient,
    truncate_data,
    prioritized_truncate_patient,
)
import random


class TestTruncationFunctions(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {
            "[CLS]": 1,
            "[SEP]": 2,
            "D_A": 3,
            "D_B": 4,
            "D_C": 5,
            "LAB_A": 6,
            "VAL_60": 7,
            "VAL_70": 8,
            "VAL_80": 9,
            "VAL_90": 10,
            "VAL_100": 11,
            "BG_Gender": 12,
        }
        self.background_len = 3
        self.sep_token = self.vocabulary.get("[SEP]")
        self.cls_token = self.vocabulary.get("[CLS]")
        self.gender_token = self.vocabulary.get("BG_Gender")

        # Sample patients
        self.sample_patient_data_1 = pd.DataFrame(
            {
                "concept": [self.cls_token, self.gender_token, self.sep_token]
                + [3] * 10
                + [self.sep_token]
                + [4] * 490
                + [self.sep_token]
                + [5] * 50
            }
        )

        self.sample_patient_data_2 = pd.DataFrame(
            {
                "concept": [self.cls_token, self.gender_token, self.sep_token]
                + [3] * 400
                + [self.sep_token]
                + [5] * 10
                + [self.sep_token]
                + [5] * 10
            }
        )

        self.sample_patient_data_3 = pd.DataFrame(
            {
                "concept": [self.cls_token, self.gender_token, self.sep_token]
                + [3] * 10
                + [self.sep_token]
                + [4] * 58
            }
        )

        self.sample_patient_data_4 = pd.DataFrame(
            {
                "concept": [self.cls_token, self.gender_token, self.sep_token]
                + [3] * 10
                + [self.sep_token]
                + [4] * 10
                + [3] * 10
                + [self.sep_token]
                + [5] * 20
                + [6 if i % 2 == 0 else random.randint(7, 11) for i in range(20)]
            }
        )

        self.sample_patient_data_5 = pd.DataFrame(
            {
                "concept": [self.cls_token, self.gender_token, self.sep_token]
                + [6 if i % 2 == 0 else random.randint(7, 11) for i in range(20)],
                "abspos": [-100, -100, -100]
                + [num for num in [i for i in range(15, 25)] for _ in range(2)],
            }
        )

        self.sample_patient_data_1["PID"] = 1
        self.sample_patient_data_2["PID"] = 2
        self.sample_patient_data_3["PID"] = 3
        self.sample_patient_data_4["PID"] = 4
        self.sample_patient_data_5["PID"] = 5

        self.sample_data = pd.concat(
            [
                self.sample_patient_data_1,
                self.sample_patient_data_2,
                self.sample_patient_data_3,
                self.sample_patient_data_4,
            ],
            ignore_index=True,
        )
        self.sample_dd_data = dd.from_pandas(self.sample_data, npartitions=1)

    def test_truncate_patient(self):
        max_len = 100
        truncated_patient = truncate_patient(
            self.sample_patient_data_1, self.background_len, max_len, self.sep_token
        )

        self.assertTrue(
            len(truncated_patient) <= max_len, "Truncated patient data exceeds max_len"
        )

    def test_truncate_patient_no_truncation_needed(self):
        max_len = 1000
        truncated_patient = truncate_patient(
            self.sample_patient_data_1, self.background_len, max_len, self.sep_token
        )
        self.assertTrue(len(truncated_patient), len(self.sample_patient_data_1))

    def test_truncate_patient_keeps_background(self):
        max_len = 100
        truncated_patient = truncate_patient(
            self.sample_patient_data_1, self.background_len, max_len, self.sep_token
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[0] == self.cls_token).any(),
            "Truncated patient data does not contain [CLS] token",
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[1] == self.gender_token).any(),
            "Truncated patient data does not contain BG token",
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[2] == self.sep_token).any(),
            "Truncated patient data does not contain [SEP] token",
        )

    def test_truncate_patient_does_not_end_with_sep(self):
        max_len = 13
        # Create a patient data where truncation might cause it to start with [SEP]
        truncated_patient = truncate_patient(
            self.sample_patient_data_3, self.background_len, max_len, self.sep_token
        )
        first_concept_after_background = truncated_patient["concept"].iloc[max_len - 1]
        self.assertNotEqual(
            first_concept_after_background,
            self.vocabulary.get("[SEP]"),
            "Truncated patient data starts with [SEP]",
        )

    def test_truncate_data_multiple_pids_truncate_all(self):
        max_len = 50
        truncated_data = truncate_data(
            self.sample_dd_data, max_len, self.vocabulary
        ).compute()

        # Verify that data is truncated correctly for each PID
        for pid in truncated_data["PID"].unique():
            pid_data_length = len(truncated_data[truncated_data["PID"] == pid])
            self.assertTrue(
                pid_data_length == max_len, f"Data for PID {pid} exceeds max_len"
            )

        # Verify that all PIDs are present after truncation
        self.assertEqual(
            len(truncated_data["PID"].unique()),
            4,
            "Not all PIDs are present after truncation",
        )

    def test_truncate_data_multiple_pids_truncate_some(self):
        max_len = 150
        truncated_data = truncate_data(
            self.sample_dd_data, max_len, self.vocabulary
        ).compute()

        # Verify that data is truncated correctly for each PID
        for pid in truncated_data["PID"].unique():
            pid_data_length = len(truncated_data[truncated_data["PID"] == pid])
            orig_len = len(self.sample_data[self.sample_data["PID"] == pid])
            if orig_len > max_len:
                self.assertTrue(
                    pid_data_length == max_len, f"Data for PID {pid} exceeds max_len"
                )
            else:
                self.assertTrue(
                    pid_data_length == orig_len,
                    f"Data for PID {pid} changed after truncation",
                )

        # Verify that all PIDs are present after truncation
        self.assertEqual(
            len(truncated_data["PID"].unique()),
            4,
            "Not all PIDs are present after truncation",
        )

    def test_truncate_patient_w_priority_drop_all(self):
        max_len = 50
        truncated_patient = prioritized_truncate_patient(
            self.sample_patient_data_4,
            self.background_len,
            max_len,
            self.sep_token,
            ["LAB", "VAL"],
            self.vocabulary,
        )
        non_priority_tokens = [
            v
            for k, v in self.vocabulary.items()
            if any(k.startswith(prefix) for prefix in ["LAB", "VAL"])
        ]

        self.assertTrue(
            len(truncated_patient) <= max_len, "Truncated patient data exceeds max_len"
        )
        self.assertTrue(
            all(
                truncated_patient["concept"].apply(
                    lambda x: x not in non_priority_tokens
                )
            )
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[0] == self.cls_token).any(),
            "Truncated patient data does not contain [CLS] token",
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[1] == self.gender_token).any(),
            "Truncated patient data does not contain BG token",
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[2] == self.sep_token).any(),
            "Truncated patient data does not contain [SEP] token",
        )

    def test_truncate_patient_w_priority_drop_some(self):
        max_len = 60
        truncated_patient = prioritized_truncate_patient(
            self.sample_patient_data_4,
            self.background_len,
            max_len,
            self.sep_token,
            ["LAB", "VAL"],
            self.vocabulary,
        )
        non_priority_tokens = [
            v
            for k, v in self.vocabulary.items()
            if any(k.startswith(prefix) for prefix in ["LAB", "VAL"])
        ]

        n_non_priority_tokens = (
            truncated_patient["concept"]
            .apply(lambda x: 1 if x in non_priority_tokens else 0)
            .sum()
        )

        self.assertEqual(5, n_non_priority_tokens)
        self.assertTrue(
            len(truncated_patient) <= max_len, "Truncated patient data exceeds max_len"
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[0] == self.cls_token).any(),
            "Truncated patient data does not contain [CLS] token",
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[1] == self.gender_token).any(),
            "Truncated patient data does not contain BG token",
        )
        self.assertTrue(
            (truncated_patient["concept"].iloc[2] == self.sep_token).any(),
            "Truncated patient data does not contain [SEP] token",
        )

    def test_truncate_patient_w_priority_unit(self):
        max_len = 8
        truncated_patient_unit = prioritized_truncate_patient(
            self.sample_patient_data_5,
            self.background_len,
            max_len,
            self.sep_token,
            ["LAB", "VAL"],
            self.vocabulary,
            unit=True,
        )

        self.assertTrue(
            len(truncated_patient_unit) == 7,
            "Truncated patient data includes sub-part of unit",
        )

    def test_truncate_patient_w_priority_no_unit(self):
        max_len = 8
        truncated_patient_no_unit = prioritized_truncate_patient(
            self.sample_patient_data_5,
            self.background_len,
            max_len,
            self.sep_token,
            ["LAB", "VAL"],
            self.vocabulary,
            unit=False,
        )

        self.assertTrue(
            len(truncated_patient_no_unit) == 8,
            "Truncated patient data has incorrect length",
        )


if __name__ == "__main__":
    unittest.main()
