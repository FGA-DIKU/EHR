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

    # Load model from the first fold
    modelmanager_trained = ModelManager(cfg, fold=None)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_decoder_model(checkpoint, [])
    
    # Debug information
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model has generate method: {hasattr(model, 'generate')}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"First patient concepts length: {len(test_dataset.patients[0].concepts)}")
    logger.info(f"First patient labels: {test_dataset.patients[0].labels}")
    
    # Simple test to see if model can predict different tokens
    logger.info("Testing model prediction capabilities...")
    test_batch = {
        "concept": torch.tensor([[1, 2, 3, 4, 5]]).to(model.device if hasattr(model, 'device') else 'cpu'),
        "segment": torch.tensor([[0, 0, 0, 0, 0]]).to(model.device if hasattr(model, 'device') else 'cpu'),
        "age": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]).to(model.device if hasattr(model, 'device') else 'cpu'),
        "abspos": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]).to(model.device if hasattr(model, 'device') else 'cpu'),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]).to(model.device if hasattr(model, 'device') else 'cpu')
    }
    
    with torch.no_grad():
        outputs = model(batch=test_batch)
        logits = outputs.logits
        last_logits = logits[:, -1, :]
        predicted_token = torch.argmax(last_logits, dim=-1).item()
        logger.info(f"Test sequence: [1, 2, 3, 4, 5]")
        logger.info(f"Predicted next token: {predicted_token}")
        logger.info(f"Expected different from 5: {predicted_token != 5}")

    gen_data = generate_sequences(cfg, test_dataset, vocab, logger, model)
    
    # Print some basic information about the generated sequences
    logger.info(f"Generated {gen_data['total_sequences']} sequences")
    if gen_data['generated_sequences']:
        first_seq = gen_data['generated_sequences'][0]
        logger.info(f"First sequence - Original length: {first_seq['original_length']}, Generated length: {first_seq['generated_length']}")
        logger.info(f"Original sequence (first 10 tokens): {first_seq['original_sequence'][:10]}")
        logger.info(f"Generated concepts (first 10 tokens): {first_seq['generated_sequence'][:10]}")
        logger.info(f"Generated segments (first 10): {first_seq['generated_segments'][:10]}")
        logger.info(f"Generated ages (first 10): {first_seq['generated_ages'][:10]}")
        logger.info(f"Generated abspos (first 10): {first_seq['generated_abspos'][:10]}")
        
        # Check if sequences are actually different
        original = first_seq['original_sequence']
        generated = first_seq['generated_sequence']
        if len(generated) > len(original):
            new_tokens = generated[len(original):]
            logger.info(f"New generated tokens: {new_tokens}")
        else:
            logger.info("No new tokens generated - sequences are identical")
        
        # Show a few more examples
        for i in range(min(3, len(gen_data['generated_sequences']))):
            seq = gen_data['generated_sequences'][i]
            logger.info(f"Sequence {i+1}: Original={seq['original_length']}, Generated={seq['generated_length']}")
            
            # Check if this sequence has new tokens
            orig = seq['original_sequence']
            gen = seq['generated_sequence']
            if len(gen) > len(orig):
                new_tokens = gen[len(orig):]
                logger.info(f"  New tokens: {new_tokens[:5]}...")  # Show first 5 new tokens
                logger.info(f"  New segments: {seq['generated_segments'][len(orig):len(orig)+5]}...")
                logger.info(f"  New ages: {seq['generated_ages'][len(orig):len(orig)+5]}...")
                logger.info(f"  New abspos: {seq['generated_abspos'][len(orig):len(orig)+5]}...")
            else:
                logger.info("  No new tokens generated")
    
    # Check if we should run sequence generation evaluation
#     run_sequence_generation_evaluation(cfg, test_dataset, vocab, folds, logger)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
