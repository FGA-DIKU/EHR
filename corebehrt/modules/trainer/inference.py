from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.modules.monitoring.logger import get_tqdm
import torch
from corebehrt.modules.trainer.shap import BEHRTWrapper
from corebehrt.modules.trainer.shap import EHRMasker
import shap
from corebehrt.modules.trainer.shap_utils import insert_concept_shap_values
from torch.utils.data import DataLoader
from corebehrt.constants.data import (
    DEFAULT_VOCABULARY, 
    MASK_TOKEN,
    CONCEPT_FEAT,
    SEGMENT_FEAT,
    AGE_FEAT,
    ABSPOS_FEAT,
    ATTENTION_MASK
)
from tqdm import tqdm

class EHRInferenceRunner(EHRTrainer):
    def _pad_sequence(self, sequence, max_len):
        pad_len = max_len - sequence.shape[0]
        if pad_len > 0:
            return torch.cat([sequence, torch.zeros(pad_len, dtype=torch.long)])
        return sequence[:max_len]

    def inference_loop(self, return_embeddings=False, shap_dict=None) -> tuple:
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        print("SHAP dict:", shap_dict)
        dataloader = self.get_dataloader(self.test_dataset, mode="test")
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description("Running inference with embeddings" if return_embeddings else "Running inference")

        logits, targets = [], []
        if return_embeddings:
            self.model.cls.eval()
            model_embs, head_embs, att_masks = [], [], []

        all_shap_values, all_inputs = [], []
        run_shap = shap_dict is not None

        if run_shap:
            max_len = shap_dict["max_len"]
            background_ids = torch.stack([
                self._pad_sequence(self.test_dataset[i]["concept"], max_len).to(torch.long)
                for i in torch.randperm(len(self.test_dataset))[:shap_dict["background_size"]]
            ]).to(self.device)

            with torch.no_grad():
                background_embeds = self.model.embeddings(
                    input_ids=background_ids,
                    segments=torch.zeros_like(background_ids),
                    age=torch.zeros_like(background_ids).float(),
                    abspos=torch.arange(background_ids.shape[1], device=self.device).unsqueeze(0).repeat(background_ids.shape[0], 1).float(),
                )

            class BatchWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    # x is now a tensor of shape [batch_size, seq_len, hidden_dim]
                    batch = {
                        "inputs_embeds": x,
                        "attention_mask": (x.abs().sum(-1) != 0).long()
                    }
                    return self.model(batch).logits

            wrapped_model = BatchWrapper(self.model)
            
            # Create background data as a single tensor
            background_data = background_embeds  # shape: [background_size, seq_len, hidden_dim]
            
            explainer = shap.GradientExplainer(
                model=wrapped_model,
                data=background_data
            )

        for batch in loop:
            self.batch_to_device(batch)

            if run_shap:
                outputs = self.model(batch)
            else:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

            logits.append(outputs.logits.float().cpu())
            targets.append(batch["target"].cpu())

            if return_embeddings:
                model_embs.append(outputs.last_hidden_state.cpu())
                head_embs.append(self.extract_head_embeddings(batch, outputs).cpu())
                att_masks.append(batch["attention_mask"].cpu())

            if run_shap:
                max_len = shap_dict["max_len"]
                input_ids = torch.stack([
                    self._pad_sequence(seq, max_len).to(torch.long)
                    for seq in batch["concept"]
                ]).to(self.device)

                with torch.no_grad():
                    input_embeds = self.model.embeddings(
                        input_ids=input_ids,
                        segments=torch.zeros_like(input_ids),
                        age=torch.zeros_like(input_ids).float(),
                        abspos=torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0).repeat(input_ids.shape[0], 1).float(),
                    )

                shap_batch_values, input_batch_ids = [], []

                for i in tqdm(range(input_embeds.shape[0]), desc="SHAP per patient"):
                    patient_embed = input_embeds[i:i+1]  # shape: [1, seq_len, hidden_dim]
                    shap_val = explainer.shap_values(patient_embed)[0][0]  # shape: [L, D]

                    token_scores = torch.from_numpy(shap_val).float().abs().sum(-1)  # shape: [L]
                    shap_batch_values.append(token_scores)
                    input_batch_ids.append(input_ids[i])

                all_shap_values.append(torch.stack(shap_batch_values))  # [B, L]
                all_inputs.append(torch.stack(input_batch_ids))         # [B, L]

        logits_tensor = torch.cat(logits, dim=0).squeeze()
        targets_tensor = torch.cat(targets, dim=0).squeeze()

        embeddings = (
            [
                torch.cat(model_embs, dim=0).squeeze(),
                torch.cat(head_embs, dim=0).squeeze(),
                torch.cat(att_masks, dim=0).squeeze(),
            ] if return_embeddings else None
        )

        if run_shap:
            all_shap_values = torch.cat(all_shap_values, dim=0)  # [N, L]
            all_inputs = torch.cat(all_inputs, dim=0)            # [N, L]
            print("✅ SHAP computed for all patients:", all_shap_values.shape[0])
            print("SHAP shape:", all_shap_values.shape, "Concept IDs shape:", all_inputs.shape)
            return logits_tensor, targets_tensor, embeddings, {
                "shap_values": all_shap_values,
                "concept_ids": all_inputs
            }

        return logits_tensor, targets_tensor, embeddings, None

    def extract_head_embeddings(self, batch, outputs):
        head_embedding = self.model.cls(
            outputs.last_hidden_state,
            attention_mask=batch["attention_mask"],
            return_embedding=True,
        )
        return head_embedding
