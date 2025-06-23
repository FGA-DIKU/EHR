from os.path import join
from typing import List, Dict, Any, Optional, Tuple
import torch
import pandas as pd
import numpy as np

from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.inference import DecoderInferenceRunner
from corebehrt.modules.preparation.dataset import DecoderDataset
from corebehrt.modules.setup.config import instantiate_function
from corebehrt.functional.evaluation.outcome_detection import (
    detect_outcome_in_sequence,
    calculate_outcome_detection_metrics
)


def inference_fold(
    finetune_folder: str,
    cfg: dict,
    test_data: DecoderDataset,
    logger,
    fold: int,
    generate_sequences: bool = False,
    generation_config: Dict[str, Any] = None
) -> Tuple[Any, Any, Any]:
    """
    Run inference for a single fold, optionally generating sequences.
    
    Args:
        finetune_folder: Path to the model folder
        cfg: Configuration dictionary
        test_data: Test dataset
        logger: Logger instance
        fold: Fold number
        generate_sequences: Whether to generate sequences instead of regular inference
        generation_config: Configuration for sequence generation
        
    Returns:
        Tuple of (predictions/probas, embeddings, generated_sequences)
    """
    fold_folder = join(finetune_folder, f"fold_{fold}")

    # Load model
    modelmanager_trained = ModelManager(cfg, fold)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_decoder_model(checkpoint, [])
    print(f"Model loaded from {fold_folder}")

    # Run inference
    return_embeddings = cfg.get("return_embeddings", False)
    evaluater = DecoderInferenceRunner(
        model=model,
        test_dataset=test_data,
        args=cfg.trainer_args,
        cfg=cfg,
    )
    
    if generate_sequences:
        # Generate sequences
        generated_sequences, _, embeddings_tensor = evaluater.inference_loop(
            return_embeddings=return_embeddings,
            generate_sequences=True,
            generation_config=generation_config
        )
        return generated_sequences, embeddings_tensor
    else:
        # Regular inference
        logits_tensor, targets_tensor, embeddings_tensor = evaluater.inference_loop(
            return_embeddings=return_embeddings
        )
        
        # Handle language modeling outputs
        if hasattr(model, 'lm_head'):
            # This is a language modeling model
            predictions, probabilities = evaluater.get_language_modeling_predictions(
                logits_tensor, targets_tensor
            )
            # For language modeling, we return predictions and probabilities
            # You might want to compute perplexity or other language modeling metrics
            return predictions.numpy(), probabilities.numpy(), embeddings_tensor
        else:
            # For other decoder models, use sigmoid for binary classification
            probas = torch.sigmoid(logits_tensor).numpy()
            return probas, embeddings_tensor


def evaluate_generated_sequences(
    generated_sequences: List[Dict[str, Any]],
    test_data: DecoderDataset,
    vocab: Dict[str, int],
    detection_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate generated sequences against real outcomes.
    
    Args:
        generated_sequences: List of generated sequences with metadata
        test_data: Test dataset containing original data
        vocab: Vocabulary dictionary
        detection_config: Configuration for outcome detection
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'total_patients': len(generated_sequences),
        'outcome_detection': {},
        'sequence_quality': {},
        'detailed_results': []
    }
    
    # Get reverse vocabulary for decoding
    id_to_token = {v: k for k, v in vocab.items()}
    
    for i, seq_data in enumerate(generated_sequences):
        # Get corresponding patient data
        patient_data = test_data.patients[seq_data['patient_index']]
        
        # Decode sequences
        original_seq = [id_to_token.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['original_sequence']]
        generated_seq = [id_to_token.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['generated_sequence']]
        
        # Use the outcome detection module
        detection_result = detect_outcome_in_sequence(
            generated_sequence=generated_seq,
            original_outcome=patient_data.outcome,
            original_abspos=patient_data.abspos,
            vocab=vocab,
            detection_config=detection_config
        )
        
        # Calculate sequence quality metrics
        sequence_metrics = calculate_sequence_quality(generated_seq, original_seq)
        
        results['detailed_results'].append({
            'pid': patient_data.pid,
            'outcome_detected': detection_result['outcome_detected'],
            'detection_method': detection_result['detection_method'],
            'detection_confidence': detection_result['confidence'],
            'detection_position': detection_result['detection_position'],
            'detection_reason': detection_result['reason'],
            'original_outcome': patient_data.outcome,
            'sequence_length': len(generated_seq),
            'sequence_quality': sequence_metrics,
            'original_sequence': ' '.join(original_seq[:20]),  # First 20 tokens
            'generated_sequence': ' '.join(generated_seq[:20])  # First 20 tokens
        })
    
    # Calculate aggregated metrics using the outcome detection module
    detection_metrics = calculate_outcome_detection_metrics(results['detailed_results'])
    results['outcome_detection'] = detection_metrics
    
    # Calculate average sequence quality metrics
    if results['detailed_results']:
        avg_quality = {}
        for key in ['perplexity', 'diversity', 'coherence']:
            values = [r['sequence_quality'].get(key, 0) for r in results['detailed_results']]
            avg_quality[key] = np.mean(values) if values else 0
        results['sequence_quality'] = avg_quality
    
    return results


def calculate_sequence_quality(generated_seq: List[str], original_seq: List[str]) -> Dict[str, float]:
    """
    Calculate quality metrics for the generated sequence.
    """
    # Calculate perplexity (simplified)
    # In practice, you'd use the model's perplexity calculation
    perplexity = 1.0  # Placeholder
    
    # Calculate diversity (unique tokens / total tokens)
    diversity = len(set(generated_seq)) / len(generated_seq) if generated_seq else 0
    
    # Calculate coherence (simplified - could be based on n-gram overlap with training data)
    coherence = 1.0  # Placeholder
    
    return {
        'perplexity': perplexity,
        'diversity': diversity,
        'coherence': coherence
    }


def get_sequence_length(dataset: DecoderDataset) -> List[int]:
    """Get sequence lengths for each patient in the dataset."""
    lengths = [len(patient.concepts) for patient in dataset.patients]
    return lengths


def compute_metrics(cfg, targets, all_probas, logger):
    """
    Computes and saves metrics for each fold and the average metrics.
    """
    if not hasattr(cfg, "metrics") or not cfg.metrics:
        return

    metrics = {k: instantiate_function(v) for k, v in cfg.metrics.items()}
    fold_metrics_list = []

    for n_fold, probas in enumerate(all_probas, start=1):
        fold_metrics = {name: func(targets, probas) for name, func in metrics.items()}
        fold_metrics["fold"] = f"fold_{n_fold}"
        fold_metrics_list.append(fold_metrics)

    metrics_df = pd.DataFrame(fold_metrics_list)
    avg_metrics = metrics_df.drop(columns=["fold"]).mean().to_dict()
    avg_metrics["fold"] = "average"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    metrics_df.to_csv(join(cfg.paths.predictions, "metrics.csv"), index=False)

    logger.info("Average metrics:")
    for key in avg_metrics:
        if key != "fold":
            logger.info(f"{key}: {avg_metrics[key]:.4f}")


def compute_sequence_generation_metrics(cfg, all_results, logger):
    """
    Computes and saves metrics for sequence generation evaluation.
    """
    if not all_results:
        return

    fold_metrics_list = []

    for n_fold, fold_results in enumerate(all_results, start=1):
        fold_metrics = {
            'fold': f'fold_{n_fold}',
            'outcome_detection_rate': fold_results['outcome_detection']['detection_rate'],
            'average_detection_confidence': fold_results['outcome_detection']['average_confidence'],
            'total_patients': fold_results['outcome_detection']['total_patients'],
            'detected_patients': fold_results['outcome_detection']['detected_patients']
        }
        
        # Add sequence quality metrics
        for key, value in fold_results['sequence_quality'].items():
            fold_metrics[f'sequence_quality_{key}'] = value
            
        fold_metrics_list.append(fold_metrics)

    metrics_df = pd.DataFrame(fold_metrics_list)
    
    # Calculate averages
    avg_metrics = metrics_df.drop(columns=["fold"]).mean().to_dict()
    avg_metrics["fold"] = "average"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    
    metrics_df.to_csv(join(cfg.paths.predictions, "sequence_generation_metrics.csv"), index=False)

    logger.info("Average sequence generation metrics:")
    for key in avg_metrics:
        if key != "fold":
            logger.info(f"{key}: {avg_metrics[key]:.4f}")

def compute_language_modeling_metrics(cfg, targets, all_predictions, all_probabilities, logger):
    """
    Computes language modeling specific metrics like perplexity.
    """
    if not hasattr(cfg, "metrics") or not cfg.metrics:
        return

    metrics = {k: instantiate_function(v) for k, v in cfg.metrics.items()}
    fold_metrics_list = []

    for n_fold, (predictions, probabilities) in enumerate(zip(all_predictions, all_probabilities), start=1):
        # Convert to tensors if they're numpy arrays
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions)
        if isinstance(probabilities, np.ndarray):
            probabilities = torch.from_numpy(probabilities)
        if isinstance(targets, np.ndarray):
            targets_tensor = torch.from_numpy(targets)
        else:
            targets_tensor = targets
            
        fold_metrics = {name: func(targets_tensor, predictions, probabilities) for name, func in metrics.items()}
        fold_metrics["fold"] = f"fold_{n_fold}"
        fold_metrics_list.append(fold_metrics)

    metrics_df = pd.DataFrame(fold_metrics_list)
    avg_metrics = metrics_df.drop(columns=["fold"]).mean().to_dict()
    avg_metrics["fold"] = "average"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    metrics_df.to_csv(join(cfg.paths.predictions, "language_modeling_metrics.csv"), index=False)

    logger.info("Average language modeling metrics:")
    for key in avg_metrics:
        if key != "fold":
            logger.info(f"{key}: {avg_metrics[key]:.4f}")

