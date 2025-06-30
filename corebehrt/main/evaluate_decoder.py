import logging
from os.path import join
import pandas as pd
import torch
import os
import numpy as np
import json

from corebehrt.constants.paths import FOLDS_FILE, PREPARED_ALL_PATIENTS
from corebehrt.constants.data import DEFAULT_VOCABULARY, EOS_TOKEN, BOS_TOKEN
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main.helper.evaluate_decoder import generate_sequences, evaluate_generated_sequences, calculate_outcome_probabilities
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
    model_cfg = load_config(join(cfg.paths.model, "train_decoder.yaml"))
    predict_all_embeddings = model_cfg.model.get("predict_all_embeddings", False)
    # cfg.sequence_generation.eos_token_id = DEFAULT_VOCABULARY[EOS_TOKEN]
    # cfg.sequence_generation.bos_token_id = DEFAULT_VOCABULARY[BOS_TOKEN]

    # Load data
    loaded_data = torch.load(
        join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS), weights_only=False
    )
    test_data = PatientDataset(loaded_data)
    outcomes = test_data.get_outcomes()
    vocab = load_vocabulary(cfg.paths.test_data_dir)
    test_dataset = DecoderDataset(test_data.patients, vocab)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.model, FOLDS_FILE), weights_only=False)
    check_for_overlap(folds, test_pids, logger)
    targets = [0 if np.isnan(x) else 1 for x in test_data.get_outcomes()]
    logger.info(f"Number of test patients: {len(test_pids)}")
    logger.info(f"Number of test positive targets: {sum(targets)}")

    # Load model from the first fold
    modelmanager_trained = ModelManager(cfg, fold=None)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_decoder_model(checkpoint, [])

    # Generate sequences
    gen_data = generate_sequences(cfg, test_dataset, vocab, logger, model, predict_all_embeddings=predict_all_embeddings)

    # Evaluate generated sequences
    target_outcomes = cfg.sequence_evaluation.outcomes
    
    # Binary evaluation
    df_results_binary, summary_metrics_binary = evaluate_generated_sequences(gen_data['generated_sequences'], test_data, vocab, target_outcomes)
    
    # Probability-based evaluation
    df_results_prob, summary_metrics_prob = calculate_outcome_probabilities(gen_data['generated_sequences'], test_data, vocab, target_outcomes)
    
    print("Binary Results:")
    print(df_results_binary)
    print("\nProbability Results:")
    print(df_results_prob)
    
    # Save evaluation results
    df_results_binary.to_csv(join(cfg.paths.predictions, "eval_gen_data_binary.csv"), index=False)
    df_results_prob.to_csv(join(cfg.paths.predictions, "eval_gen_data_probabilities.csv"), index=False)
    
    # Save summary metrics
    with open(join(cfg.paths.predictions, "summary_metrics_binary.json"), 'w') as f:
        json.dump(summary_metrics_binary, f, indent=2)
    
    with open(join(cfg.paths.predictions, "summary_metrics_probabilities.json"), 'w') as f:
        json.dump(summary_metrics_prob, f, indent=2)
    
    print("\nBinary Summary Metrics:")
    print(summary_metrics_binary)
    print("\nProbability Summary Metrics:")
    print(summary_metrics_prob)
    print(f"\nDetailed results saved to: {join(cfg.paths.predictions, 'eval_gen_data_*.csv')}")

    # Check if we should run sequence generation evaluation
#     run_sequence_generation_evaluation(cfg, test_dataset, vocab, folds, logger)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
