from os.path import join
from typing import List, Dict, Any, Optional, Tuple
import torch
import pandas as pd
import numpy as np

from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.inference import DecoderInferenceRunner
from corebehrt.modules.preparation.dataset import DecoderDataset
from corebehrt.modules.setup.config import instantiate_function


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


def generate_sequences(
    cfg: dict,
    test_dataset: DecoderDataset,
    vocab: Dict[str, int],
    logger,
    model: torch.nn.Module
) -> Dict[str, Any]:
    """
    Generate sequences using the decoder model.
    
    Args:
        cfg: Configuration dictionary
        test_dataset: Test dataset
        vocab: Vocabulary dictionary
        logger: Logger instance
        model: Pre-loaded model to use for generation
        
    Returns:
        Dictionary containing generated sequences and metadata
    """
    logger.info("Starting sequence generation")
    
    # Get generation configuration
    generation_config = cfg.get("sequence_generation", {
        'max_length': 512,
        'num_beams': 1,
        'do_sample': True,  # Enable sampling
        'temperature': 0.8,  # Lower temperature for some randomness
        'top_p': 0.9,  # Use nucleus sampling
        'pad_token_id': 0,
        'eos_token_id': 2,
        'bos_token_id': 1
    })
    
    # Create inference runner
    evaluater = DecoderInferenceRunner(
        model=model,
        test_dataset=test_dataset,
        args=cfg.trainer_args,
        cfg=cfg,
    )
    
    # Generate sequences
    generated_sequences, _ = evaluater.inference_loop(
        return_embeddings=False,
        generate_sequences=True,
        generation_config=generation_config
    )
    
    # Prepare results
    results = {
        'generated_sequences': generated_sequences,
        'total_sequences': len(generated_sequences),
        'generation_config': generation_config,
        'vocab_size': len(vocab)
    }
    
    logger.info(f"Sequence generation completed. Total sequences: {len(generated_sequences)}")
    
    return results

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
        
        # # Use the outcome detection module
        # detection_result = detect_outcome_in_sequence(
        #     generated_sequence=generated_seq,
        #     original_outcome=patient_data.outcome,
        #     original_abspos=patient_data.abspos,
        #     vocab=vocab,
        #     detection_config=detection_config
        # )
        
        # Calculate basic sequence quality metrics
        sequence_metrics = {
            'length': len(generated_seq),
            'unique_tokens': len(set(generated_seq)),
            'vocabulary_coverage': len(set(generated_seq)) / len(vocab) if len(vocab) > 0 else 0
        }
        
        results['detailed_results'].append({
            'pid': patient_data.pid,
            # 'outcome_detected': detection_result['outcome_detected'],
            # 'detection_method': detection_result['detection_method'],
            # 'detection_confidence': detection_result['confidence'],
            # 'detection_position': detection_result['detection_position'],
            # 'detection_reason': detection_result['reason'],
            'original_outcome': patient_data.outcome,
            'sequence_length': len(generated_seq),
            'sequence_quality': sequence_metrics,
            'original_sequence': ' '.join(original_seq[:20]),  # First 20 tokens
            'generated_sequence': ' '.join(generated_seq[:20])  # First 20 tokens
        })
    
    # Calculate aggregated metrics using the outcome detection module
    # detection_metrics = calculate_outcome_detection_metrics(results['detailed_results'])
    # results['outcome_detection'] = detection_metrics
    
    # # Calculate average sequence quality metrics
    # if results['detailed_results']:
    #     avg_quality = {}
    #     for key in ['perplexity', 'diversity', 'coherence']:
    #         values = [r['sequence_quality'].get(key, 0) for r in results['detailed_results']]
    #         avg_quality[key] = np.mean(values) if values else 0
    #     results['sequence_quality'] = avg_quality
    
    return results
