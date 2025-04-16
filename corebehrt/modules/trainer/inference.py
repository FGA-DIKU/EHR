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
        loop.set_description("Running inference")

        logits_list = []
        targets_list = []
        if return_embeddings:
            bi_gru = self.model.cls
            bi_gru.eval()
            embeddings_list = []
            loop.set_description("Running inference with embeddings")

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

                if return_embeddings:
                    last_hidden_state = outputs.last_hidden_state
                    attention_mask = batch["attention_mask"]
                    with torch.no_grad():
                        embedding = bi_gru(
                            last_hidden_state,
                            attention_mask=attention_mask,
                            return_embedding=True,
                        )
                        embeddings_list.append(embedding.cpu())

                logits_list.append(
                    outputs.logits.float().cpu()
                )  # .float to convert to float32 (from bfloat16)
                targets_list.append(batch["target"].cpu())

        logits_tensor = torch.cat(logits_list, dim=0).squeeze()
        targets_tensor = torch.cat(targets_list, dim=0).squeeze()

        if return_embeddings:
            embeddings_tensor = torch.cat(embeddings_list, dim=0).squeeze()
            print(f"Embeddings shape: {embeddings_tensor.shape}")
        else:
            embeddings_tensor = None

        return logits_tensor, targets_tensor, embeddings_tensor
