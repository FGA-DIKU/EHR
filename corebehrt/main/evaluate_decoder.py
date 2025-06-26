import logging
from os.path import join
import pandas as pd
import torch
import os
import numpy as np

from corebehrt.constants.paths import FOLDS_FILE, PREPARED_ALL_PATIENTS
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main.helper.evaluate_decoder import generate_sequences
from corebehrt.modules.preparation.dataset import DecoderDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.modules.setup.manager import ModelManager

CONFIG_PATH = "./corebehrt/configs/evaluate_decoder.yaml"


def main_evaluate(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate_decoder()

    # Logger
    logger = logging.getLogger("evaluate")

    # Setup config
    cfg.trainer_args = {}
    batch_size_value = cfg.get("test_batch_size", 128)
    for key in ["test_batch_size", "effective_batch_size", "batch_size"]:
        cfg.trainer_args[key] = batch_size_value
    cfg.paths.restart_model = cfg.paths.model

    # Load data
    loaded_data = torch.load(
        join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS), weights_only=False
    )
    test_data = PatientDataset(loaded_data)
    vocab = load_vocabulary(cfg.paths.test_data_dir)
    test_dataset = DecoderDataset(test_data.patients, vocab)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.model, FOLDS_FILE), weights_only=False)
    check_for_overlap(folds, test_pids, logger)
    targets = [0 if np.isnan(x) else 1 for x in test_data.get_outcomes()]
    logger.info(f"Number of test patients: {len(test_pids)}")
    logger.info(f"Number of test positive targets: {sum(targets)}")

    # Load model
    modelmanager_trained = ModelManager(cfg, fold=None)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_decoder_model(checkpoint, [])
    

    # gen_data = generate_sequences(cfg, test_dataset, vocab, folds, logger)
    
    # # Print some basic information about the generated sequences
    # logger.info(f"Generated {gen_data['total_sequences']} sequences across {gen_data['num_folds']} folds")
    # if gen_data['generated_sequences']:
    #     first_seq = gen_data['generated_sequences'][0]
    #     logger.info(f"First sequence - Original length: {first_seq['original_length']}, Generated length: {first_seq['generated_length']}")
    
    # Check if we should run sequence generation evaluation
#     run_sequence_generation_evaluation(cfg, test_dataset, vocab, folds, logger)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
