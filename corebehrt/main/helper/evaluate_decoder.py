from typing import List, Dict, Any
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from corebehrt.modules.preparation.dataset import DecoderDataset, PatientDataset
from corebehrt.constants.data import DEFAULT_VOCABULARY, EOS_TOKEN

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def _get_generation_config(cfg: dict) -> dict:
    """
    Get the generation configuration from the config.
    """
    # Get generation configuration
    generation_config = cfg.get("sequence_generation")
    if generation_config is None:
        raise ValueError("sequence_generation configuration is required but not found in config")
    
    # Check for required parameters
    required_params = ['max_length', 'do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty']
    missing_params = [param for param in required_params if param not in generation_config]
    if missing_params:
        raise ValueError(f"Missing required generation parameters: {missing_params}")
    
    # Add eos_token_id if not present
    if 'eos_token_id' not in generation_config:
        generation_config['eos_token_id'] = DEFAULT_VOCABULARY[EOS_TOKEN]
    
    return generation_config

def generate_sequences(
    cfg: dict,
    test_dataset: DecoderDataset,
    vocab: Dict[str, int],
    logger,
    model: torch.nn.Module,
    predict_all_embeddings: bool = False
) -> Dict[str, Any]:
    """
    Generate sequences using the decoder model.
    
    Args:
        cfg: Configuration dictionary
        test_dataset: Test dataset
        vocab: Vocabulary dictionary
        logger: Logger instance
        model: Pre-loaded model to use for generation
        predict_all_embeddings: Whether to predict all embeddings or just concepts
        
    Returns:
        Dictionary containing generated sequences and metadata
    """
    logger.info("Starting sequence generation")
    
    # Get generation configuration
    generation_config = _get_generation_config(cfg)
    
    # Get dataloader
    from corebehrt.modules.trainer.trainer import EHRTrainer
    trainer = EHRTrainer(
        model=model,
        test_dataset=test_dataset,
        args=cfg.trainer_args,
        cfg=cfg,
    )
    dataloader = trainer.get_dataloader(test_dataset, mode="test")
    
    generated_sequences = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            trainer.batch_to_device(batch)
            
            # Generate sequences using the model's generate method
            generated_outputs = model.generate(
                batch, 
                predict_all_embeddings=predict_all_embeddings,
                **generation_config
            )
            
            # Store generated sequences with metadata
            batch_size = batch["concept"].size(0)
            for i in range(batch_size):
                seq_data = {
                    'patient_index': batch_idx * cfg.trainer_args.get('batch_size', 32) + i,
                    'original_sequence': batch["concept"][i].cpu().tolist(),
                    'generated_sequence': generated_outputs['concepts'][i].cpu().tolist(),
                    'original_length': batch["concept"][i].size(0),
                    'generated_length': generated_outputs['concepts'][i].size(0),
                }
                
                # Only include additional embeddings if predict_all_embeddings is True
                if predict_all_embeddings:
                    seq_data.update({
                        'generated_segments': generated_outputs['segments'][i].cpu().tolist(),
                        'generated_ages': generated_outputs['ages'][i].cpu().tolist(),
                        'generated_abspos': generated_outputs['abspos'][i].cpu().tolist()
                    })
                
                generated_sequences.append(seq_data)
    
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
    test_data: PatientDataset,
    vocab: Dict[str, int],
    outcomes: List[str]
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate generated sequences against real outcomes.
    
    Args:
        generated_sequences: List of generated sequences with metadata
        outcomes: List of outcomes to evaluate against
        
    Returns:
        Tuple of (DataFrame with detailed results, Dictionary with summary metrics)
    """
    
    detailed_results = []
    
    for i, seq_data in enumerate(generated_sequences):
        # Get corresponding patient data
        patient_index = int(seq_data['patient_index'])  # Ensure it's an integer
        patient_data = test_data.patients[patient_index]
        
        # Decode sequences
        original_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['original_sequence']]
        generated_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['generated_sequence']]
        
        target_outcome_detected = 1 if any(outcome in generated_seq for outcome in outcomes) else 0
        
        # Calculate basic sequence quality metrics
        sequence_metrics = {
            'length': len(generated_seq),
            'unique_tokens': len(set(generated_seq)),
            'vocabulary_coverage': len(set(generated_seq)) / len(vocab) if len(vocab) > 0 else 0
        }
        
        detailed_results.append({
            'pid': patient_data.pid,
            'target_outcome_detected': target_outcome_detected,
            'original_outcome': 1 if not pd.isna(patient_data.outcome) else 0,
            'sequence_length': len(generated_seq),
            'sequence_quality_length': sequence_metrics['length'],
            'sequence_quality_unique_tokens': sequence_metrics['unique_tokens'],
            'sequence_quality_vocabulary_coverage': sequence_metrics['vocabulary_coverage'],
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(detailed_results)
    
    # Calculate summary metrics
    summary_metrics = {}
    if len(df_results) > 0:
        total_sequences = len(df_results)
        target_outcomes_detected = df_results['target_outcome_detected'].sum()
        
        # Collect original outcomes and detected outcomes for accuracy calculation
        original_outcomes = df_results['original_outcome'].tolist()
        detected_outcomes = df_results['target_outcome_detected'].tolist()
        
        # Calculate TP, TN, FP, FN
        tp = sum(1 for orig, det in zip(original_outcomes, detected_outcomes) if orig == 1 and det == 1)
        tn = sum(1 for orig, det in zip(original_outcomes, detected_outcomes) if orig == 0 and det == 0)
        fp = sum(1 for orig, det in zip(original_outcomes, detected_outcomes) if orig == 0 and det == 1)
        fn = sum(1 for orig, det in zip(original_outcomes, detected_outcomes) if orig == 1 and det == 0)
        
        summary_metrics = {
            'total_sequences': total_sequences,
            'target_outcomes_detected': target_outcomes_detected,
            'target_outcome_detection_rate': target_outcomes_detected / total_sequences if total_sequences > 0 else 0,
            'accuracy': accuracy_score(original_outcomes, detected_outcomes),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        # Convert numpy types to native Python types for JSON serialization
        summary_metrics = convert_numpy_types(summary_metrics)
    
    return df_results, summary_metrics

def calculate_outcome_probabilities(
    generated_sequences: List[Dict[str, Any]],
    test_data: PatientDataset,
    vocab: Dict[str, int],
    outcomes: List[str]
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate outcome probabilities from generated sequences.
    
    Args:
        generated_sequences: List of generated sequences with metadata
        outcomes: List of outcomes to evaluate against
        
    Returns:
        Tuple of (DataFrame with detailed results, Dictionary with summary metrics)
    """
    
    detailed_results = []
    
    for i, seq_data in enumerate(generated_sequences):
        # Get corresponding patient data
        patient_index = int(seq_data['patient_index'])  # Ensure it's an integer
        patient_data = test_data.patients[patient_index]
        
        # Decode sequences
        original_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['original_sequence']]
        generated_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['generated_sequence']]
        
        # Calculate probability of each outcome appearing in the sequence
        outcome_probabilities = {}
        for outcome in outcomes:
            # Count occurrences of the outcome in the generated sequence
            outcome_count = generated_seq.count(outcome)
            # Calculate probability as count / sequence length
            outcome_prob = outcome_count / len(generated_seq) if len(generated_seq) > 0 else 0.0
            outcome_probabilities[outcome] = outcome_prob
        
        # Calculate overall outcome probability (max probability across all outcomes)
        max_outcome_prob = max(outcome_probabilities.values()) if outcome_probabilities else 0.0
        
        # Calculate basic sequence quality metrics
        sequence_metrics = {
            'length': len(generated_seq),
            'unique_tokens': len(set(generated_seq)),
            'vocabulary_coverage': len(set(generated_seq)) / len(vocab) if len(vocab) > 0 else 0
        }
        
        result_row = {
            'pid': patient_data.pid,
            'original_outcome': 1 if not pd.isna(patient_data.outcome) else 0,
            'sequence_length': len(generated_seq),
            'sequence_quality_length': sequence_metrics['length'],
            'sequence_quality_unique_tokens': sequence_metrics['unique_tokens'],
            'sequence_quality_vocabulary_coverage': sequence_metrics['vocabulary_coverage'],
            'max_outcome_probability': max_outcome_prob,
        }
        
        # Add individual outcome probabilities
        for outcome in outcomes:
            result_row[f'prob_{outcome}'] = outcome_probabilities[outcome]
        
        detailed_results.append(result_row)
    
    # Create DataFrame
    df_results = pd.DataFrame(detailed_results)
    
    # Calculate summary metrics
    summary_metrics = {}
    if len(df_results) > 0:
        total_sequences = len(df_results)
        
        # Calculate AUC using probabilities
        from sklearn.metrics import roc_auc_score
        original_outcomes = df_results['original_outcome'].tolist()
        outcome_probs = df_results['max_outcome_probability'].tolist()
        
        # Calculate AUC (handle case where all outcomes are same class)
        try:
            auc_score = roc_auc_score(original_outcomes, outcome_probs)
        except ValueError:
            auc_score = 0.5  # Default AUC when all outcomes are same class
        
        # Calculate binary predictions for confusion matrix
        binary_predictions = [1 if prob > 0.0 else 0 for prob in outcome_probs]
        
        # Calculate TP, TN, FP, FN
        tp = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == 1 and pred == 1)
        tn = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == 0 and pred == 0)
        fp = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == 0 and pred == 1)
        fn = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == 1 and pred == 0)
        
        summary_metrics = {
            'total_sequences': total_sequences,
            'auc_score': auc_score,
            'mean_outcome_probability': df_results['max_outcome_probability'].mean(),
            'std_outcome_probability': df_results['max_outcome_probability'].std(),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': (tp + tn) / total_sequences if total_sequences > 0 else 0
        }
        
        # Convert numpy types to native Python types for JSON serialization
        summary_metrics = convert_numpy_types(summary_metrics)
    
    return df_results, summary_metrics
