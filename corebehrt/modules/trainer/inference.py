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
            return_embeddings: Whether to return embeddings
            generation_config: Configuration for sequence generation
            
        Returns:
            Tuple of (generated_sequences, embeddings)
        """
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        dataloader = self.get_dataloader(self.test_dataset, mode="test")
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description("Generating sequences")

        generated_sequences = []
        if return_embeddings:
            model_embs, head_embs, att_masks = [], [], []

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
                input_ids = batch.get("input_ids", batch.get("input_sequence"))
                attention_mask = batch.get("attention_mask")
                
                if input_ids is None:
                    raise ValueError("No input_ids or input_sequence found in batch")
                
                # Generate sequences
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    generated_outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=generation_config.get('max_length', 512),
                        num_beams=generation_config.get('num_beams', 1),
                        do_sample=generation_config.get('do_sample', False),
                        temperature=generation_config.get('temperature', 1.0),
                        top_p=generation_config.get('top_p', 1.0),
                        pad_token_id=generation_config.get('pad_token_id', 0),
                        eos_token_id=generation_config.get('eos_token_id', 2),
                        bos_token_id=generation_config.get('bos_token_id', 1),
                        early_stopping=True
                    )
                
                # Store generated sequences with metadata
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    seq_data = {
                        'patient_index': batch_idx * self.args.get('batch_size', 32) + i,
                        'original_sequence': input_ids[i].cpu().tolist(),
                        'generated_sequence': generated_outputs[i].cpu().tolist(),
                        'original_length': input_ids[i].size(0),
                        'generated_length': generated_outputs[i].size(0)
                    }
                    generated_sequences.append(seq_data)
                
                # Store embeddings if requested
                if return_embeddings:
                    # Get embeddings from the model
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    model_emb = outputs.last_hidden_state
                    att = attention_mask
                    
                    # Extract head embeddings if available
                    if hasattr(self.model, 'cls'):
                        head_emb = self.extract_head_embeddings(batch, outputs)
                    else:
                        head_emb = model_emb[:, 0, :]  # Use CLS token embedding
                    
                    model_embs.append(model_emb.cpu())
                    head_embs.append(head_emb.cpu())
                    att_masks.append(att.cpu())

        # Prepare embeddings output
        embeddings = None
        if return_embeddings:
            embeddings = [
                torch.cat(model_embs, dim=0).squeeze(),
                torch.cat(head_embs, dim=0).squeeze(),
                torch.cat(att_masks, dim=0).squeeze(),
            ]

        return generated_sequences, embeddings

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


