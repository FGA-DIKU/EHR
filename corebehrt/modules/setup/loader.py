import logging
from os.path import join

import torch
from transformers import ModernBertConfig

from corebehrt.functional.setup.model import get_last_checkpoint_epoch

logger = logging.getLogger(__name__)  # Get the logger for this module

CHECKPOINT_FOLDER = "checkpoints"


class ModelLoader:
    def __init__(self, model_path: str, checkpoint_epoch: str = None):
        """Load model from config and checkpoint."""
        self.model_path = model_path
        self.checkpoint_epoch = checkpoint_epoch

    def load_model(
        self, model_class, add_config: dict = {}, checkpoint: dict = None, kwargs={}
    ):
        """Load model from config and checkpoint. model_class is the class of the model to be loaded."""
        checkpoint = self.load_checkpoint() if checkpoint is None else checkpoint
        # Load the config from file
        config = ModernBertConfig.from_pretrained(self.model_path)
        config.update(add_config)
        model = model_class(config, **kwargs)

        return self.load_state_dict_into_model(model, checkpoint)

    def load_state_dict_into_model(
        self, model: torch.nn.Module, checkpoint: dict
    ) -> torch.nn.Module:
        """Load state dict into model. If embeddings are not loaded, raise an error."""
        load_result = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        missing_keys = load_result.missing_keys

        if len([k for k in missing_keys if k.startswith("embeddings")]) > 0:
            pretrained_model_embeddings = model.embeddings.__class__.__name__
            raise ValueError(
                f"Embeddings not loaded. Ensure that model.behrt_embeddings is compatible with pretrained model embeddings {pretrained_model_embeddings}."
            )
        logger.warning("missing state dict keys: %s", missing_keys)
        return model

    def load_checkpoint(self) -> dict:
        """Load checkpoint, if checkpoint epoch provided. Else load last checkpoint."""
        checkpoints_dir = join(self.model_path, CHECKPOINT_FOLDER)
        checkpoint_epoch = self.get_checkpoint_epoch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = join(
            checkpoints_dir, f"checkpoint_epoch{checkpoint_epoch}_end.pt"
        )
        logger.info("Loading checkpoint from %s", checkpoint_path)
        return torch.load(checkpoint_path, map_location=device, weights_only=False)

    def get_checkpoint_epoch(self) -> int:
        """Get checkpoint if set or return the last checkpoint_epoch for this model."""
        if self.checkpoint_epoch is None:
            logger.info("No checkpoint provided. Loading last checkpoint.")
            self.checkpoint_epoch = get_last_checkpoint_epoch(
                join(self.model_path, CHECKPOINT_FOLDER)
            )
        return self.checkpoint_epoch
