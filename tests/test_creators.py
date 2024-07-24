import unittest
import pandas as pd
from datetime import datetime

from corebehrt.functional.creators import (
    create_ages,
    create_abspos,
    create_segments,
    create_background,
    create_death,
)


class TestCreators(unittest.TestCase):
    def setUp(self):
        self.origin_point = datetime(2020, 1, 26)
        self.background_vars = ["GENDER"]

        self.concepts = pd.DataFrame(
            {
                "PID": ["1", "2", "3", "1"],
                "CONCEPT": ["DA1", "DA2", "MA1", "DA2"],
                "TIMESTAMP": pd.to_datetime(
                    ["2020-01-02", "2021-03-20", "2022-05-08", "2023-01-02"]
                ),
                "ADMISSION_ID": ["A", "B", "C", "D"],
            }
        )
        self.patients_info = pd.DataFrame(
            {
                "PID": ["1", "2", "3"],
                "BIRTHDATE": pd.to_datetime(["2000-01-02", "2000-03-20", "2000-05-08"]),
                "GENDER": ["Male", "Female", "Male"],
            }
        )

    def test_age_creator(self):
        # Aligned birth dates
        birthdates = (
            self.patients_info.set_index("PID")
            .loc[self.concepts.PID]
            .reset_index()
            .BIRTHDATE
        )
        result = create_ages(self.concepts.TIMESTAMP, birthdates)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.map(round).tolist(), [20, 21, 22, 23])

    def test_abspos_creator(self):
        result = create_abspos(self.concepts.TIMESTAMP, self.origin_point)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(
            result.tolist(),
            (
                (self.concepts.TIMESTAMP - self.origin_point).dt.total_seconds()
                / 60
                / 60
            ).tolist(),
        )

    def test_segment_creator(self):
        groups = self.concepts.groupby("PID")["ADMISSION_ID"]
        result = create_segments(groups)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.tolist(), [0, 0, 0, 1])

    def test_background_creator(self):
        result = create_background(self.patients_info, self.background_vars)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["PID", "concept", "TIMESTAMP"])

        # Check that all added concept are prefixed by BG_
        self.assertTrue(result.concept.str.startswith("BG_").all())

        # Check, that each patient has one of each background_vars
        for _, patient in result.groupby("PID"):
            self.assertEqual(len(patient), len(self.background_vars))
            patient_vars = patient.concept.apply(lambda s: s.split("_")[1])
            self.assertEqual(list(patient_vars), self.background_vars)


if __name__ == "__main__":
    unittest.main()
