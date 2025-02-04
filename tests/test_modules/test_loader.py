import os
import shutil
import unittest
from tempfile import mkdtemp

import pandas as pd

from corebehrt.modules.loader import FormattedDataLoader


class TestFormattedDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = mkdtemp()

        # Create test patients_info data
        patients_info_data = pd.DataFrame(
            {
                "PID": [1, 2],
                "BIRTHDATE": pd.to_datetime(["2000-01-01", "1990-01-01"]),
                "DEATHDATE": pd.to_datetime(["2020-01-01", "2010-01-01"]),
            }
        )
        patients_info_path = os.path.join(self.test_dir, "patients_info.parquet")
        patients_info_data.to_parquet(patients_info_path)

        # Create test concept data
        concept_data = pd.DataFrame(
            {
                "PID": [1, 2],
                "TIMESTAMP": pd.to_datetime(["2020-01-01", "2010-01-01"]),
                "CONCEPT": ["diagnosis", "medication"],
                "ADMISSION_ID": [1, 2],
            }
        )
        concept_path = os.path.join(self.test_dir, "concept.diagnosis.parquet")
        concept_data.to_parquet(concept_path)

    def tearDown(self):
        # Remove the temporary directory and files
        shutil.rmtree(self.test_dir)

    def test_load(self):
        # Initialize the FormattedDataLoader
        loader = FormattedDataLoader(folder=self.test_dir, concept_types=["diagnosis"])

        # Load the data
        concepts, patients_info = loader.load()

        # Convert Dask DataFrames to Pandas DataFrames for testing
        concepts_df = concepts.compute()
        patients_info_df = patients_info.compute()

        # Verify the patients_info data
        expected_patients_info_data = pd.DataFrame(
            {
                "PID": [1, 2],
                "BIRTHDATE": pd.to_datetime(["2000-01-01", "1990-01-01"]),
                "DEATHDATE": pd.to_datetime(["2020-01-01", "2010-01-01"]),
            }
        )
        pd.testing.assert_frame_equal(patients_info_df, expected_patients_info_data)

        # Verify the concepts data
        expected_concepts_data = pd.DataFrame(
            {
                "PID": [1, 2],
                "TIMESTAMP": pd.to_datetime(["2020-01-01", "2010-01-01"]),
                "CONCEPT": pd.Series(
                    ["diagnosis", "medication"], dtype="string[pyarrow]"
                ),
                "ADMISSION_ID": [1, 2],
            }
        )
        pd.testing.assert_frame_equal(concepts_df, expected_concepts_data)


if __name__ == "__main__":
    unittest.main()
