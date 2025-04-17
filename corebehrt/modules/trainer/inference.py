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
            ehr_embs, model_embs, head_embs = [], [], []

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

                if return_embeddings:
                    ehr, model, head = self.extract_embeddings(batch, outputs)
                    ehr_embs.append(ehr.cpu())
                    model_embs.append(model.cpu())
                    head_embs.append(head.cpu())

                logits.append(outputs.logits.float().cpu())
                targets.append(batch["target"].cpu())

        logits_tensor = torch.cat(logits, dim=0).squeeze()
        targets_tensor = torch.cat(targets, dim=0).squeeze()

        embeddings = (
            [
                torch.cat(ehr_embs, dim=0).squeeze(),
                torch.cat(model_embs, dim=0).squeeze(),
                torch.cat(head_embs, dim=0).squeeze(),
            ]
            if return_embeddings
            else None
        )

        return logits_tensor, targets_tensor, embeddings

    def extract_embeddings(self, batch, outputs):
        ehr_embedding = self._get_ehr_embedding(batch)
        model_embedding = self._mean_pool(
            outputs.last_hidden_state, batch["attention_mask"]
        )
        head_embedding = self.model.cls(
            outputs.last_hidden_state,
            attention_mask=batch["attention_mask"],
            return_embedding=True,
        )
        return ehr_embedding, model_embedding, head_embedding

    def _get_ehr_embedding(self, batch):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            ehr_output = self.model(batch, return_embeddings=True)
        return ehr_output.mean(dim=1)

    def _mean_pool(self, hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        summed = torch.sum(hidden_state * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / count
