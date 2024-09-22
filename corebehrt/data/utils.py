import os
import re
import numpy as np
import logging
from typing import Dict, List, Tuple

from corebehrt.common.utils import Data

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = 'PID'
BG_GENDER_KEYS = {
    'male': ['M', 'Mand',  'male', 'Male', 'man', 'MAN', '1'],
    'female': ['W', 'Kvinde', 'F', 'female', 'Female', 'woman', 'WOMAN', '0']
}
MIN_POSITIVES = {'finetune': 1, None: 1}
CHECKPOINT_FOLDER = 'checkpoints'
ORIGIN_POINT = {'year': 2020, 'month': 1, 'day': 26, 'hour': 0, 'minute': 0, 'second': 0}


class Utilities:
    @classmethod
    def process_data(cls, data: Data, func: callable, args_for_func: Dict={})->Dict:
        """Apply a function to all datasets in a dictionary"""
        data = func(data, **args_for_func)

        return data

    @staticmethod
    def get_last_checkpoint_epoch(checkpoint_folder: str)->int:
        """Returns the epoch of the last checkpoint."""
        # Regular expression to match the pattern retry_XXX
        pattern = re.compile(r"checkpoint_epoch(\d+)_end\.pt$")
        max_epoch = None
        for filename in os.listdir(checkpoint_folder):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                if max_epoch is None or epoch > max_epoch:
                    max_epoch = epoch
        # Return the folder with the maximum retry number
        if max_epoch is None:
            raise ValueError("No checkpoint found in folder {}".format(checkpoint_folder))

        return max_epoch


    @staticmethod
    def select_and_reorder_feats_and_pids(feats: Dict[str, List], pids: List[str], select_pids: List[str])->Tuple[Dict[str, List], List[str]]:
        """Reorders pids and feats to match the order of select_pids"""
        if not set(select_pids).issubset(set(pids)):
            raise ValueError(f"There are {len(set(select_pids).difference(set(pids)))} select pids not present in pids") 
        pid2idx = {pid: index for index, pid in enumerate(pids)}
        indices_to_keep = [pid2idx[pid] for pid in select_pids] # order is important, so keep select_pids as list
        selected_feats = {}
        for key, value in feats.items():
            selected_feats[key] = [value[idx] for idx in indices_to_keep]
        return selected_feats, select_pids

    @staticmethod
    def calculate_ages_at_censor_date(data: Data) -> List[int]:
        """
        Calculates the age of patients at their respective censor dates.
        """
        ages_at_censor_date = []
        
        for abspos, age, censor_date in zip(data.features['abspos'], data.features['age'], data.censor_outcomes):
            if censor_date is None:
                ages_at_censor_date.append(age[-1]) # if no censoring, we take the last age
                continue
            # Calculate age differences and find the closest abspos index to the censor date
            time_differences_h = np.array([censor_date - ap for ap in abspos])
            # compute closest index (with regards to abspos) on the left to censor date
            closest_abspos_index = np.argmin(
                np.where(time_differences_h < 0, np.inf, time_differences_h)) 
            age_at_censor = age[closest_abspos_index] + time_differences_h[closest_abspos_index] / 24 / 365.25
            ages_at_censor_date.append(age_at_censor)
        return ages_at_censor_date

