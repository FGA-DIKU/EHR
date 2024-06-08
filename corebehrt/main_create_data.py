"""
Input: Formatted Data
- Load concepts
- Handle wrong data
- Exclude patients with <k concepts
- Split data
- Tokenize
- truncate train and val
"""
import os
import shutil
from os.path import join

import torch
from tqdm import tqdm

from corebehrt.classes.excluder import Excluder
# New stuff
from corebehrt.classes.features import FeatureCreator
from corebehrt.common.azure import AzurePathContext, save_to_blobstore
from corebehrt.common.config import load_config
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.common.utils import check_directory_for_features
from corebehrt.data.batch import Batches, BatchTokenize
from corebehrt.data.concept_loader import ConceptLoaderLarge
from corebehrt.data.tokenizer import EHRTokenizer
from common.logger import TqdmToLogger

CONFIG_NAME = 'create_data.yaml'
BLOBSTORE = 'PHAIR'

args = get_args(CONFIG_NAME, 'create_data')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_data(config_path):
    """
        Loads data
        Finds outcomes
        Creates features
        Handles wrong data
        Excludes patients with <k concepts
        Splits data
        Tokenizes
        Saves
    """
    cfg = load_config(config_path)

    cfg, _, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).azure_data_pretrain_setup()

    logger = DirectoryPreparer(config_path).prepare_directory(cfg)  
    logger.info('Mount Dataset')
    
    logger.info('Initialize Processors')
    logger.info('Starting feature creation and processing')
    if not check_directory_for_features(cfg.loader.data_dir):
        pids = create_and_save_features(ConceptLoaderLarge(**cfg.loader), 
                                        Excluder(**cfg.excluder), # Excluder is the new Handler and old Excluder
                                        cfg, logger)
        torch.save(pids, join(cfg.output_dir, 'features', 'pids_features.pt'))
    else:
        pids = torch.load(join(cfg.loader.data_dir, 'features', 'pids_features.pt'))
    logger.info('Finished feature creation and processing')
    
    logger.info('Splitting batches')
    batches = Batches(cfg, pids)
    batches_split = batches.split_batches()
    
    tokenized_dir_name = cfg.get('tokenized_dir_name','tokenized')
    check_and_clear_directory(cfg, logger, tokenized_dir_name=tokenized_dir_name)
    
    vocabulary = None
    if 'vocabulary' in cfg.paths:
        logger.info(f'Loading vocabulary from {cfg.paths.vocabulary}')
        vocabulary = torch.load(cfg.paths.vocabulary) 

    logger.info('Tokenizing')
    tokenizer = EHRTokenizer(config=cfg.tokenizer, vocabulary=vocabulary)
    batch_tokenize = BatchTokenize(pids, tokenizer, cfg, tokenized_dir_name=tokenized_dir_name)
    shutil.copy(config_path, join(cfg.output_dir, tokenized_dir_name,  'data_cfg.yaml'))
    
    batch_tokenize.tokenize(batches_split)
    logger.info('Finished tokenizing')
    
    if cfg.env=='azure':
        features_dir_name  = cfg.paths.get('save_features_dir_name', cfg.paths.run_name)
        save_to_blobstore(local_path='data/', 
                          remote_path=join(BLOBSTORE, 'features', features_dir_name))
        mount_context.stop()
    logger.info('Finished')

def check_and_clear_directory(cfg, logger, tokenized_dir_name='tokenized'):
    tokenized_dir = join(cfg.output_dir, tokenized_dir_name)
    tokenized_files = os.listdir(tokenized_dir) 
    if len(tokenized_files)>0:
        logger.warning(f"The directory {tokenized_dir} is not empty.")
        logger.warning(f"Deleting tokenized files.")
        for file in tokenized_files:
            file_path = join(tokenized_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)

def create_and_save_features(conceptloader:ConceptLoaderLarge, 
                             excluder: Excluder, 
                             cfg, logger)-> list:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    pids = []
    for i, (concept_batch, patient_batch) in enumerate(tqdm(conceptloader(), desc='Batch Process Data', file=TqdmToLogger(logger))):
        feature_creator = FeatureCreator(cfg.features) # Otherwise appended to old features
        concept_batch = feature_creator(concept_batch, patient_batch)
        excluder = Excluder(**cfg.excluder)
        concept_batch = excluder.exclude_incorrect_events(concept_batch)
        concept_batch, pids = excluder.exclude_short_sequences(concept_batch)
        # write to disk e.g. parquet
        concept_batch.to_csv(join(cfg.output_dir, 'features', f'features.csv'), index=False, mode='a' if i > 0 else 'w')

        #torch.save(features_batch, join(cfg.output_dir, 'features', f'features_{i}.pt'))
        #torch.save(pids_batch, join(cfg.output_dir, 'features', f'pids_features_{i}.pt'))
        #pids.append(pids_batch)
    assert False
    return pids


if __name__ == '__main__':
    main_data(config_path)


