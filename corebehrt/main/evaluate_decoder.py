import logging
from os.path import join
import pandas as pd
import torch
import os

from corebehrt.constants.paths import FOLDS_FILE, PREPARED_ALL_PATIENTS
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.evaluate_decoder import (
    inference_fold, 
    compute_metrics, 
    evaluate_generated_sequences,
    compute_sequence_generation_metrics
)
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.modules.preparation.dataset import DecoderDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.config import instantiate_function
from corebehrt.functional.io_operations.load import load_vocabulary

CONFIG_PATH = "./corebehrt/configs/evaluate_decoder.yaml"


def main_evaluate(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate()

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
    vocab = load_vocabulary(cfg.paths.prepared_data)
    test_dataset = DecoderDataset(test_data.patients, vocab)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.folds_dir, FOLDS_FILE), weights_only=False)
    check_for_overlap(folds, test_pids, logger)
    targets = test_data.get_outcomes()
    logger.info(f"Number of test patients: {len(test_pids)}")
    logger.info(f"Number of test positive targets: {sum(targets)}")

    # Check if we should run sequence generation evaluation
    run_sequence_generation = cfg.get("run_sequence_generation", False)
    
    if run_sequence_generation:
        logger.info("Running sequence generation evaluation...")
        run_sequence_generation_evaluation(cfg, test_dataset, vocab, folds, logger)
    else:
        logger.info("Running regular inference evaluation...")
        run_regular_inference_evaluation(cfg, test_dataset, test_pids, targets, folds, logger)


def run_sequence_generation_evaluation(cfg, test_dataset, vocab, folds, logger):
    """Run sequence generation evaluation."""
    
    # Get generation and detection configuration
    generation_config = cfg.get("sequence_generation", {})
    detection_config = cfg.get("outcome_detection", {})
    
    all_results = []
    
    for n_fold, fold in enumerate(folds, start=1):
        logger.info(f"Evaluating fold {n_fold} for sequence generation")
        
        # Generate sequences
        generated_sequences, embeddings = inference_fold(
            finetune_folder=cfg.paths.model,
            cfg=cfg,
            test_data=test_dataset,
            logger=logger,
            fold=n_fold,
            generate_sequences=True,
            generation_config=generation_config
        )
        
        # Evaluate generated sequences
        fold_results = evaluate_generated_sequences(
            generated_sequences=generated_sequences,
            test_data=test_dataset,
            vocab=vocab,
            detection_config=detection_config
        )
        fold_results['fold'] = n_fold
        all_results.append(fold_results)
        
        # Save detailed results for this fold
        detailed_df = pd.DataFrame(fold_results['detailed_results'])
        detailed_df.to_csv(join(cfg.paths.predictions, f"fold_{n_fold}_sequence_generation_detailed.csv"), index=False)
        
        # Save some example generated sequences
        examples_df = pd.DataFrame([
            {
                'pid': r['pid'],
                'original_sequence': r['original_sequence'],
                'generated_sequence': r['generated_sequence'],
                'outcome_detected': r['outcome_detected'],
                'detection_method': r['detection_method'],
                'detection_confidence': r['detection_confidence'],
                'original_outcome': r['original_outcome']
            }
            for r in fold_results['detailed_results'][:10]  # First 10 examples
        ])
        examples_df.to_csv(join(cfg.paths.predictions, f"fold_{n_fold}_sequence_generation_examples.csv"), index=False)
        
        logger.info(f"Fold {n_fold} - Outcome detection rate: {fold_results['outcome_detection']['detection_rate']:.4f}")

    # Compute and save metrics
    compute_sequence_generation_metrics(cfg, all_results, logger)
    
    logger.info("Sequence generation evaluation completed!")


def run_regular_inference_evaluation(cfg, test_dataset, test_pids, targets, folds, logger):
    """Run regular inference evaluation."""
    
    # Get predictions
    combined_df = pd.DataFrame(
        {
            "pid": test_pids,
            "target": targets,
        }
    )
    if cfg.get("save_info", False):
        for k, v in cfg.save_info.items():
            func = instantiate_function(v)
            combined_df[k] = func(test_dataset)

    all_probas = []
    for n_fold, fold in enumerate(folds, start=1):
        probas, embeddings = inference_fold(
            finetune_folder=cfg.paths.model,
            cfg=cfg,
            test_data=test_dataset,
            logger=logger,
            fold=n_fold,
            generate_sequences=False
        )
        combined_df[f"fold_{n_fold}_probas"] = probas
        all_probas.append(probas)

        # Save embeddings if specified
        if embeddings is not None:
            save_emb_path = join(cfg.paths.predictions, "embeddings", f"fold_{n_fold}")
            os.makedirs(save_emb_path, exist_ok=True)
            torch.save(embeddings[0], join(save_emb_path, "BERT_embeddings.pt"))
            torch.save(embeddings[1], join(save_emb_path, "cls_embeddings.pt"))
            torch.save(embeddings[2], join(save_emb_path, "attention_masks.pt"))
            torch.save(test_pids, join(save_emb_path, "pids.pt"))

    # Save combined predictions and metrics if specified
    combined_df.to_csv(join(cfg.paths.predictions, "predictions.csv"), index=False)
    compute_metrics(cfg, targets, all_probas, logger)
    
    logger.info("Regular inference evaluation completed!")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
