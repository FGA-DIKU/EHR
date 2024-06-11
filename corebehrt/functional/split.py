from typing import Tuple
import numpy as np

def split_pids_into_pt_ft_test(pids: list, pretrain:float, finetune:float, test:float)->Tuple[list, list, list]:
    if not np.isclose(pretrain + finetune + test, 1.0):
        raise ValueError("The sum of pretrain, finetune, and test should be 1.")
    n = len(pids)
    pretrain_pids = pids[:int(n*pretrain)]
    finetune_pids = pids[int(n*pretrain):int(n*(pretrain+finetune))]
    test_pids = pids[int(n*(pretrain+finetune)):]
    return pretrain_pids, finetune_pids, test_pids