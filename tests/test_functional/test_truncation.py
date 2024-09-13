import unittest
import pandas as pd
import dask.dataframe as dd
from corebehrt.functional.utils import truncate_patient, truncate_data

class TestTruncationFunctions(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {"[CLS]": 1, "[SEP]": 2, "A": 3, "B": 4, "C": 5}
        self.background_len = 2
        self.sep_token = self.vocabulary.get("[SEP]")
        self.cls_token = self.vocabulary.get("[CLS]")
        
        # Sample patients
        self.sample_patient_data_1 = pd.DataFrame({
            "concept": [self.cls_token, self.sep_token] + [3] * 10 + [self.sep_token] + [4] * 490 + [self.sep_token] + [5] * 50
        })

        self.sample_patient_data_2 = pd.DataFrame({
            "concept": [self.cls_token, self.sep_token] + [3] * 400 + [self.sep_token] + [5] * 10 + [self.sep_token] + [5] * 10
        })

        self.sample_patient_data_3 = pd.DataFrame({
            "concept": [self.cls_token, self.sep_token] + [3] * 10 + [self.sep_token] + [4] * 58
        })

        self.sample_patient_data_1['PID'] = 1
        self.sample_patient_data_2['PID'] = 2
        self.sample_patient_data_3['PID'] = 3

        self.sample_data = pd.concat([self.sample_patient_data_1, self.sample_patient_data_2, self.sample_patient_data_3], ignore_index=True)
        self.sample_dd_data = dd.from_pandas(self.sample_data, npartitions=1)
    
    def test_truncate_patient(self):
        max_len = 100
        truncated_patient = truncate_patient(self.sample_patient_data_1, self.background_len, max_len, self.vocabulary)

        self.assertTrue(len(truncated_patient) <= max_len, "Truncated patient data exceeds max_len")

    def test_truncate_patient_no_truncation_needed(self):
        max_len = 1000
        truncated_patient = truncate_patient(self.sample_patient_data_1, self.background_len, max_len, self.vocabulary)
        self.assertTrue(len(truncated_patient), len(self.sample_patient_data_1))

    def test_truncate_patient_keeps_background(self):
        max_len = 100
        truncated_patient = truncate_patient(self.sample_patient_data_1, self.background_len, max_len, self.vocabulary)
        self.assertTrue((truncated_patient["concept"].iloc[0] == self.cls_token).any(), "Truncated patient data does not contain [CLS] token")
        self.assertTrue((truncated_patient["concept"].iloc[1] == self.sep_token).any(), "Truncated patient data does not contain [SEP] token")

    def test_truncate_patient_does_not_end_with_sep(self):
        max_len = 13
        # Create a patient data where truncation might cause it to start with [SEP]
        truncated_patient = truncate_patient(self.sample_patient_data_3, self.background_len, max_len, self.vocabulary)
        first_concept_after_background = truncated_patient["concept"].iloc[max_len - 1]
        self.assertNotEqual(first_concept_after_background, self.vocabulary.get("[SEP]"), "Truncated patient data starts with [SEP]")

    def test_truncate_data_multiple_pids_truncate_all(self):
        max_len = 50
        truncated_data = truncate_data(self.sample_dd_data, max_len, self.vocabulary).compute()
        
        # Verify that data is truncated correctly for each PID
        for pid in truncated_data['PID'].unique():
            pid_data_length = len(truncated_data[truncated_data['PID'] == pid])
            self.assertTrue(pid_data_length == max_len, f"Data for PID {pid} exceeds max_len")

        # Verify that all PIDs are present after truncation
        self.assertEqual(len(truncated_data['PID'].unique()), 3, "Not all PIDs are present after truncation")

    def test_truncate_data_multiple_pids_truncate_some(self):
        max_len = 150
        truncated_data = truncate_data(self.sample_dd_data, max_len, self.vocabulary).compute()
        
        # Verify that data is truncated correctly for each PID
        for pid in truncated_data['PID'].unique():
            pid_data_length = len(truncated_data[truncated_data['PID'] == pid])
            orig_len = len(self.sample_data[self.sample_data['PID'] == pid])
            if orig_len > max_len:
                self.assertTrue(pid_data_length == max_len, f"Data for PID {pid} exceeds max_len")
            else:
                self.assertTrue(pid_data_length == orig_len, f"Data for PID {pid} changed after truncation")

        # Verify that all PIDs are present after truncation
        self.assertEqual(len(truncated_data['PID'].unique()), 3, "Not all PIDs are present after truncation")


if __name__ == '__main__':
    unittest.main()