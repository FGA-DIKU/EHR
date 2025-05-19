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

        for batch in loop:
            self.batch_to_device(batch)

            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = self.model(batch)

            logits.append(outputs.logits.float().cpu())
            targets.append(batch["target"].cpu())

            if return_embeddings:
                model_embs.append(outputs.last_hidden_state.cpu())
                head_embs.append(self.extract_head_embeddings(batch, outputs).cpu())
                att_masks.append(batch["attention_mask"].cpu())

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
            shap_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

            all_shap_values = []
            all_inputs = []
            all_indexes = []

            for batch in shap_dataloader:
                wrapped_model = BEHRTWrapper(self.model, batch)

                # Get background as float concept IDs
                background_data = wrapped_model.get_background_data(shap_dict.get("background_size", 100))

                # Initialize SHAP explainer
                explainer = shap.GradientExplainer(wrapped_model, background_data)

                # Get the current input (concept IDs only)
                input_tensor = batch[CONCEPT_FEAT].float().to(self.model.device)

                # Compute SHAP values
                shap_values, indexes = explainer.shap_values(input_tensor, ranked_outputs=1)

                # Fix shapes: shap_values[0] is [1, seq_len] → squeeze to [seq_len]
                shap_tensor = torch.tensor(shap_values[0]).squeeze(0)
                concept_tensor = batch[CONCEPT_FEAT].squeeze(0)  # [seq_len]

                all_shap_values.append(shap_tensor)
                all_inputs.append(concept_tensor)
                all_indexes.append(torch.tensor(indexes))

            # Combine across patients (i.e., batch dimension)
            shap_results = {
                'shap_values': all_shap_values,     # list of [seq_len]
                'concept_ids': all_inputs,          # list of [seq_len]
                'indexes': all_indexes              # list of [1]
            }

            return logits_tensor, targets_tensor, embeddings, shap_results

        return logits_tensor, targets_tensor, embeddings, None

    def extract_head_embeddings(self, batch, outputs):
        head_embedding = self.model.cls(
            outputs.last_hidden_state,
            attention_mask=batch["attention_mask"],
            return_embedding=True,
        )
        return head_embedding
