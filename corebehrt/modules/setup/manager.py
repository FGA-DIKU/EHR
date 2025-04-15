import logging
import os
from os.path import join

from corebehrt.functional.setup.model import (
    get_last_checkpoint_epoch,
    load_model_cfg_from_checkpoint,
)
from corebehrt.modules.setup.config import Config
from corebehrt.modules.setup.directory import CHECKPOINTS_DIR
from corebehrt.modules.setup.initializer import Initializer
from corebehrt.modules.setup.loader import ModelLoader

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for initializing model, optimizer and scheduler."""

    def __init__(self, cfg: Config, fold: int):
        self.cfg = cfg
        self.fold = fold

        # Determine where to load existing model from. Load from:
        # -> model_path if it has checkpoints and is readable.
        # -> restart_model_path if it is set and has checkpoints.
        # -> pretrain_model_path if no current model exist at the other paths.

        self.model_path = self.check_model("model", fold=fold, checkpoints=False)
        self.pretrain_model_path = self.check_model("pretrain_model")

        # Restart model from other directory (if given)
        self.restart_model_path = self.check_model("restart_model", fold=fold)
        cfg_path = self.cfg.paths.get("restart_model")

        # Update config from old model, if relevant
        if self.restart_model_path is not None:
            self.cfg.model = load_model_cfg_from_checkpoint(cfg_path, "finetune_config")

        # Check arguments are valid
        self.check_arguments()

        self.checkpoint_model_path = self.restart_model_path or self.pretrain_model_path
        logger.info(f"Checkpoint model path: {self.checkpoint_model_path}")
        self.initializer = None

    def check_arguments(self):
        if self.model_path is None:
            raise ValueError("Model path must be set!")

        if (
            self.pretrain_model_path is None
            and self.restart_model_path is None
            and not self.check_checkpoints(self.model_path)
        ):
            raise ValueError(
                "Either paths.pretrain_model or paths.restart_model must be set, if no existing model is provided."
            )

    def check_model(
        self, model: str, fold: int = None, checkpoints: bool = True
    ) -> str:
        if not hasattr(self.cfg.paths, model):
            return None

        path = self.cfg.paths[model]
        if fold is not None:
            path = join(path, f"fold_{fold}")

        if not os.path.exists(path):
            logger.warning(f"Could not find model at path '{path}'.")
            return None

        if checkpoints and not self.check_checkpoints(path):
            logger.warning(f"No checkpoints found at path '{path}'.")
            return None

        return path

    @staticmethod
    def check_checkpoints(path: str) -> bool:
        return path is not None and len(os.listdir(join(path, CHECKPOINTS_DIR))) > 0

    def load_checkpoint(self, checkpoints=False):
        if checkpoints and self.check_checkpoints(self.model_path):
            checkpoint_path = self.model_path
        else:
            checkpoint_path = self.checkpoint_model_path

        return ModelLoader(checkpoint_path).load_checkpoint()

    def initialize_finetune_model(self, checkpoint, outcomes):
        logger.info("Initializing model")
        self.initializer = Initializer(
            self.cfg, checkpoint=checkpoint, model_path=self.checkpoint_model_path
        )
        model = self.initializer.initialize_finetune_model(outcomes)
        return model

    def initialize_training_components(self, model, outcomes):
        """Initialize training components. If no model_path provided, optimizer and scheduler are initialized from scratch."""
        if self.restart_model_path is None:
            logger.info("Initializing optimizer and scheduler from scratch")
            self.initializer.checkpoint = None
        optimizer = self.initializer.initialize_optimizer(model)
        sampler, cfg = self.initializer.initialize_sampler(outcomes)
        scheduler = self.initializer.initialize_scheduler(optimizer)
        return optimizer, sampler, scheduler, cfg

    def get_epoch(self):
        """Get epoch from model_path."""
        if self.restart_model_path is None:
            return 0
        else:
            return get_last_checkpoint_epoch(
                join(self.restart_model_path, CHECKPOINTS_DIR)
            )
