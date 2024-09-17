import dask.dataframe as dd
import logging
from os.path import join
import torch
import os 

logger = logging.getLogger(__name__)

def save_sequence_lengths(data: dd.DataFrame, save_dir: str, desc:str='') -> None:
    """
        Save the number of rows for each PID in a list.
        Assumes that the data has a column named PID.
        Optional: Save using a description.
    """
    os.makedirs(save_dir, exist_ok=True)
    sequence_lens = data.groupby("PID").size().compute()
    sequence_lens_list = sequence_lens.tolist()
    torch.save(sequence_lens_list, join(save_dir, f'sequences_lengths{desc}.pt'))

def save_data(data: dd.DataFrame, vocabulary: dict, save_dir: str, desc: str = '', outcomes: dd.DataFrame = None) -> None:
    """
    Save data and vocabulary to a folder.
    Optionally save outcomes DataFrame if provided.
    Optional: Save using a description.
    """
    os.makedirs(save_dir, exist_ok=True)
    data.to_csv(join(save_dir, f'data{desc}', '*.csv'), index=False)
    torch.save(vocabulary, join(save_dir, f'vocabulary.pt'))
    if outcomes is not None:
        outcomes.to_csv(join(save_dir, f'outcomes{desc}', '*.csv'), index=False)

def save_pids_splits(train_data: dd.DataFrame, val_data: dd.DataFrame, save_dir: str) -> None:
    """
    Save train and val data to a folder.
    Assumes that the data has a column named PID.
    """
    os.makedirs(save_dir, exist_ok=True)
    train_pids = train_data['PID'].unique().compute().tolist()
    val_pids = val_data['PID'].unique().compute().tolist()
    torch.save(train_pids, join(save_dir, 'pids_train.pt'))
    torch.save(val_pids, join(save_dir, 'pids_val.pt'))