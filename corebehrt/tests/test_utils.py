import unittest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from data.utils import Utilities

class TestUtilities(unittest.TestCase):
    def test_process_datasets(self):
        data = MagicMock()
        data.pids = [1, 2, 3]
        func = Mock(side_effect=lambda x: x)
        func.__name__ = "func"
        Utilities.process_data(data, func)
        func.assert_called_once_with(data)

    def test_log_patient_nums(self):
        pass

    def test_select_and_order_outcomes_for_patients(self):
        all_outcomes = {'PID': [3, 2, 1], "COVID": [1, 2, 0]}
        pids = [1, 2, 3]
        outcome = "COVID"
        outcomes = Utilities.select_and_order_outcomes_for_patients(all_outcomes, pids, outcome)

        self.assertEqual(outcomes, [0, 2, 1])

    def test_get_abspos_from_origin_point(self):
        timestamps = [datetime(1, 1, i) for i in range(1,11)]
        origin_point = {'year': 1, 'month': 1, 'day': 1}
        abspos = Utilities.get_abspos_from_origin_point(timestamps, origin_point)

        self.assertEqual(abspos, [24*i for i in range(10)])

    def test_check_and_adjust_max_segment(self):
        data = Mock(features={'segment': [[1,2,3], [4,5,6]]})
        model_config = Mock(type_vocab_size=5)
        Utilities.check_and_adjust_max_segment(data, model_config)

        self.assertEqual(model_config.type_vocab_size, 7)

    def test_get_gender_token(self):
        BG_GENDER_KEYS = {
            'male': ['M', 'Mand',  'male', 'Male', 'man', 'MAN', '1'],
            'female': ['W', 'Kvinde', 'F', 'female', 'Female', 'woman', 'WOMAN', '0']
        }
        vocabulary = {'BG_GENDER_Male': 0, 'BG_GENDER_Female': 1}
        for gender, values in BG_GENDER_KEYS.items():
            for value in values:
                result = Utilities.get_gender_token(vocabulary, value)
                if gender == "male":
                    self.assertEqual(result, 0)
                elif gender == "female":
                    self.assertEqual(result, 1)

    def test_log_pos_patients_num(self):
        pass

    @patch('os.listdir', return_value=[f"checkpoint_epoch{i}_end.pt" for i in range(10)])
    def test_get_last_checkpoint_epoch(self, mock_listdir):
        last_epoch = Utilities.get_last_checkpoint_epoch("dir")

        self.assertEqual(last_epoch, 9)

    def test_iter_patients(self):
        pass


if __name__ == '__main__':
    unittest.main()