from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.modules.monitoring.logger import get_tqdm
import torch
from corebehrt.modules.trainer.shap import BEHRTWrapper
from corebehrt.modules.trainer.shap import EHRMasker
import shap
from corebehrt.modules.trainer.shap_utils import insert_concept_shap_values
from torch.utils.data import DataLoader

class EHRInferenceRunner(EHRTrainer):
    def inference_loop(self, return_embeddings=False, run_shap=True) -> tuple:
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

        all_shap_values = None
        if run_shap:
            patient_shap_vals = []
            patient_concept_ids = []

            masker = EHRMasker()
            shap_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
            for batch in shap_dataloader:
                wrapped_model = BEHRTWrapper(self.model, batch)
                concepts = batch["concept"].numpy().reshape(-1, 1, batch["concept"].shape[1])
                explainer = shap.PermutationExplainer(wrapped_model, masker=masker, batch_size=16)
                n_permutations = concepts.shape[-1] * 2 + 1
                shap_values = explainer.shap_values(concepts, npermutations=n_permutations)

                # Convert to torch tensors
                patient_shap_vals.append(torch.tensor(shap_values[0][0]))   # [seq_len]
                patient_concept_ids.append(torch.tensor(concepts[0][0]))    # [seq_len]

            comb_shap_vals = {
                "shap_values": patient_shap_vals,   # list of [seq_len] tensors
                "concept_ids": patient_concept_ids  # list of [seq_len] tensors
            }

        return logits_tensor, targets_tensor, embeddings, comb_shap_vals

    def extract_head_embeddings(self, batch, outputs):
        head_embedding = self.model.cls(
            outputs.last_hidden_state,
            attention_mask=batch["attention_mask"],
            return_embedding=True,
        )
        return head_embedding
