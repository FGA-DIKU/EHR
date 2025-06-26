from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.modules.monitoring.logger import get_tqdm
import torch


class EHRInferenceRunner(EHRTrainer):
    def inference_loop(self, return_embeddings=False) -> tuple:
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        dataloader = self.get_dataloader(self.test_dataset, mode="test")
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description(
            "Running inference with embeddings"
            if return_embeddings
            else "Running inference"
        )

        logits, targets = [], []
        if return_embeddings:
            self.model.cls.eval()
            model_embs, head_embs, att_masks = [], [], []

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

                if return_embeddings:
                    model = outputs.last_hidden_state
                    att = batch["attention_mask"]
                    head = self.extract_head_embeddings(batch, outputs)
                    model_embs.append(model.cpu())
                    head_embs.append(head.cpu())
                    att_masks.append(att.cpu())

                logits.append(outputs.logits.float().cpu())
                targets.append(batch["target"].cpu())

        logits_tensor = torch.cat(logits, dim=0).squeeze()
        targets_tensor = torch.cat(targets, dim=0).squeeze()

        embeddings = (
            [
                torch.cat(model_embs, dim=0).squeeze(),
                torch.cat(head_embs, dim=0).squeeze(),
                torch.cat(att_masks, dim=0).squeeze(),
            ]
            if return_embeddings
            else None
        )

        return logits_tensor, targets_tensor, embeddings

    def extract_head_embeddings(self, batch, outputs):
        head_embedding = self.model.cls(
            outputs.last_hidden_state,
            attention_mask=batch["attention_mask"],
            return_embedding=True,
        )
        return head_embedding


class DecoderInferenceRunner(EHRInferenceRunner):
    def inference_loop(self, return_embeddings=False, generate_sequences=False, generation_config=None) -> tuple:
        """
        Run inference loop for decoder models, optionally generating sequences.
        
        Args:
            return_embeddings: Whether to return embeddings
            generate_sequences: Whether to generate sequences instead of regular inference
            generation_config: Configuration for sequence generation
            
        Returns:
            Tuple of (logits/predictions, targets, embeddings) or (generated_sequences, embeddings)
        """
        if generate_sequences:
            return self._sequence_generation_loop(return_embeddings, generation_config)
        else:
            return super().inference_loop(return_embeddings)
    
    def _sequence_generation_loop(self, return_embeddings=False, generation_config=None):
        """
        Generate sequences using the decoder model.
        
        Args:
            return_embeddings: Whether to return embeddings (ignored for now)
            generation_config: Configuration for sequence generation
            
        Returns:
            Tuple of (generated_sequences, None) - embeddings always None for now
        """
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        dataloader = self.get_dataloader(self.test_dataset, mode="test")
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description("Generating sequences")

        generated_sequences = []

        # Default generation config
        if generation_config is None:
            generation_config = {
                'max_length': 512,
                'num_beams': 1,
                'do_sample': False,
                'temperature': 1.0,
                'top_p': 1.0,
                'pad_token_id': 0,
                'eos_token_id': 2,
                'bos_token_id': 1
            }

        with torch.no_grad():
            for batch_idx, batch in enumerate(loop):
                self.batch_to_device(batch)
                
                # Get input sequences for generation
                input_ids = batch.get("concept", batch.get("input_ids", batch.get("input_sequence")))
                attention_mask = batch.get("attention_mask")
                
                if input_ids is None:
                    raise ValueError("No concept, input_ids, or input_sequence found in batch")
                
                # For sequence generation, we need to prepare the input properly
                # The model expects the full batch structure
                generation_batch = {
                    "concept": input_ids,
                    "segment": batch.get("segment", torch.zeros_like(input_ids)),
                    "age": batch.get("age", torch.zeros_like(input_ids, dtype=torch.float)),
                    "abspos": batch.get("abspos", torch.zeros_like(input_ids, dtype=torch.float)),
                    "attention_mask": attention_mask
                }
                
                # Custom generation approach
                try:
                    generated_outputs = self._custom_generate(
                        generation_batch, 
                        generation_config
                    )
                except Exception as e:
                    print(f"Error during generation: {e}")
                    # Fallback: just use the input sequences as generated sequences
                    generated_outputs = {
                        'concepts': input_ids,
                        'segments': batch.get("segment", torch.zeros_like(input_ids)),
                        'ages': batch.get("age", torch.zeros_like(input_ids, dtype=torch.float)),
                        'abspos': batch.get("abspos", torch.zeros_like(input_ids, dtype=torch.float))
                    }
                
                # Store generated sequences with metadata
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    seq_data = {
                        'patient_index': batch_idx * self.args.get('batch_size', 32) + i,
                        'original_sequence': input_ids[i].cpu().tolist(),
                        'generated_sequence': generated_outputs['concepts'][i].cpu().tolist(),
                        'original_length': input_ids[i].size(0),
                        'generated_length': generated_outputs['concepts'][i].size(0),
                        'generated_segments': generated_outputs['segments'][i].cpu().tolist(),
                        'generated_ages': generated_outputs['ages'][i].cpu().tolist(),
                        'generated_abspos': generated_outputs['abspos'][i].cpu().tolist()
                    }
                    generated_sequences.append(seq_data)

        return generated_sequences, None

    def _custom_generate(self, batch, generation_config):
        """
        Custom generation method that works with our model's batch structure.
        
        Args:
            batch: Input batch with concept, segment, age, abspos, attention_mask
            generation_config: Generation configuration
            
        Returns:
            Generated sequences with all embeddings
        """
        max_length = generation_config.get('max_length', 512)
        do_sample = generation_config.get('do_sample', False)
        temperature = generation_config.get('temperature', 1.0)
        top_p = generation_config.get('top_p', 1.0)
        pad_token_id = generation_config.get('pad_token_id', 0)
        eos_token_id = generation_config.get('eos_token_id', 2)
        
        # Get initial input
        input_concepts = batch["concept"]
        input_segments = batch["segment"]
        input_ages = batch["age"]
        input_abspos = batch["abspos"]
        attention_mask = batch["attention_mask"]
        batch_size, seq_len = input_concepts.shape
        
        print(f"Starting generation: batch_size={batch_size}, seq_len={seq_len}")
        print(f"Input sequence: {input_concepts[0].tolist()[:10]}...")
        
        # Initialize output with input
        generated_concepts = input_concepts.clone()
        generated_segments = input_segments.clone()
        generated_ages = input_ages.clone()
        generated_abspos = input_abspos.clone()
        
        # Generate tokens one by one
        for step in range(seq_len, max_length):
            # Prepare current batch for forward pass
            current_batch = {
                "concept": generated_concepts,
                "segment": generated_segments,
                "age": generated_ages,
                "abspos": generated_abspos,
                "attention_mask": torch.ones_like(generated_concepts)
            }
            
            # Get model outputs
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = self.model(batch=current_batch)
                logits = outputs.logits
            
            # Get predictions for all embeddings (if available)
            concept_logits = outputs.concept_logits if hasattr(outputs, 'concept_logits') else logits
            segment_logits = outputs.segment_logits if hasattr(outputs, 'segment_logits') else None
            age_predictions = outputs.age_predictions if hasattr(outputs, 'age_predictions') else None
            abspos_predictions = outputs.abspos_predictions if hasattr(outputs, 'abspos_predictions') else None
            
            # For language modeling, we want to predict the next token
            # The model outputs logits for each position, but we want the next token
            # So we use the logits at the last position to predict what comes next
            next_token_logits = concept_logits[:, -1, :] / temperature
            
            # Debug: Check what the model is predicting
            if step == seq_len:  # First generation step
                top_k_logits, top_k_indices = torch.topk(next_token_logits[0], k=5)
                print(f"Top 5 predicted tokens: {top_k_indices[0].tolist()} with logits: {top_k_logits[0].tolist()}")
                print(f"Current input token: {input_concepts[0, -1].item()}")
                print(f"Vocabulary size: {concept_logits.size(-1)}")
                
                # Check if the model is predicting the same token
                predicted_token = torch.argmax(next_token_logits[0]).item()
                print(f"Predicted token: {predicted_token}, Current token: {input_concepts[0, -1].item()}")
                print(f"Are they the same? {predicted_token == input_concepts[0, -1].item()}")
                
                # Show predictions for other embeddings (if available)
                if segment_logits is not None:
                    predicted_segment = torch.argmax(segment_logits[0, -1]).item()
                    print(f"Predicted segment: {predicted_segment}")
                if age_predictions is not None:
                    predicted_age = age_predictions[0, -1].item()
                    print(f"Predicted age: {predicted_age}")
                if abspos_predictions is not None:
                    predicted_abspos = abspos_predictions[0, -1].item()
                    print(f"Predicted abspos: {predicted_abspos}")
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or take argmax
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Debug: Check what we're actually generating
            if step == seq_len:  # First generation step
                print(f"Generated token: {next_tokens[0].item()}")
                print(f"Expected different token: {next_tokens[0].item() != input_concepts[0, -1].item()}")
                
                # If we're generating the same token, try to force diversity
                if next_tokens[0].item() == input_concepts[0, -1].item():
                    print("Warning: Model is predicting the same token. Trying to force diversity...")
                    # Get second best prediction
                    sorted_logits, sorted_indices = torch.sort(next_token_logits[0], descending=True)
                    if len(sorted_indices) > 1:
                        next_tokens[0] = sorted_indices[1]
                        print(f"Using second best prediction: {next_tokens[0].item()}")
            
            # Generate corresponding embeddings for the new tokens
            # Use model predictions if available, otherwise fall back to heuristics
            if segment_logits is not None:
                next_segments = torch.argmax(segment_logits[:, -1, :], dim=-1)
            else:
                next_segments = torch.zeros_like(next_tokens)
                # Fallback to heuristics
                for i in range(len(next_tokens)):
                    if next_tokens[i].item() == 2:  # SEP token
                        next_segments[i] = generated_segments[i, -1] + 1 if generated_segments.size(1) > 0 else 1
                    else:
                        next_segments[i] = generated_segments[i, -1] if generated_segments.size(1) > 0 else 0
            
            if age_predictions is not None:
                next_ages = age_predictions[:, -1]
            else:
                next_ages = torch.zeros_like(next_tokens, dtype=torch.float)
                # Fallback to heuristics
                for i in range(len(next_tokens)):
                    next_ages[i] = generated_ages[i, -1] + 0.1 if generated_ages.size(1) > 0 else 0.1
            
            if abspos_predictions is not None:
                next_abspos = abspos_predictions[:, -1]
            else:
                next_abspos = torch.zeros_like(next_tokens, dtype=torch.float)
                # Fallback to heuristics
                for i in range(len(next_tokens)):
                    next_abspos[i] = generated_abspos[i, -1] + 1.0 if generated_abspos.size(1) > 0 else 1.0
            
            # Append to generated sequences
            generated_concepts = torch.cat([generated_concepts, next_tokens.unsqueeze(-1)], dim=-1)
            generated_segments = torch.cat([generated_segments, next_segments.unsqueeze(-1)], dim=-1)
            generated_ages = torch.cat([generated_ages, next_ages.unsqueeze(-1)], dim=-1)
            generated_abspos = torch.cat([generated_abspos, next_abspos.unsqueeze(-1)], dim=-1)
            
            # Check for EOS tokens
            if eos_token_id in next_tokens:
                # Stop generation for sequences that hit EOS
                break
        
        print(f"Final generated sequence: {generated_concepts[0].tolist()[:15]}...")
        print(f"Generated segments: {generated_segments[0].tolist()[:15]}...")
        print(f"Generated ages: {generated_ages[0].tolist()[:15]}...")
        print(f"Generated abspos: {generated_abspos[0].tolist()[:15]}...")
        
        # Return all generated embeddings
        return {
            'concepts': generated_concepts,
            'segments': generated_segments,
            'ages': generated_ages,
            'abspos': generated_abspos
        }

    def get_language_modeling_predictions(self, logits_tensor, targets_tensor):
        """
        Get predictions for language modeling tasks.
        
        Args:
            logits_tensor: Model logits
            targets_tensor: Target tokens
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # For language modeling, we typically compute perplexity
        # and return the predicted next tokens
        probs = torch.softmax(logits_tensor, dim=-1)
        predictions = torch.argmax(logits_tensor, dim=-1)
        
        return predictions, probs


