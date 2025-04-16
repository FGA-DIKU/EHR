from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.modules.monitoring.logger import get_tqdm
import torch 

class EHRInferenceRunner(EHRTrainer):
    def inference_loop(self) -> tuple:
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        dataloader = self.get_dataloader(self.test_dataset, mode="test")
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description("Running inference")

        logits_list = []
        targets_list = []

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

                logits_list.append(
                    outputs.logits.float().cpu()
                )  # .float to convert to float32 (from bfloat16)
                targets_list.append(batch["target"].cpu())
        
        logits_tensor = torch.cat(logits_list, dim=0).squeeze()
        targets_tensor = torch.cat(targets_list, dim=0).squeeze()

        return logits_tensor, targets_tensor
