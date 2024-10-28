import logging
import os
from os.path import join
from typing import Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import Sampler
from transformers import BertConfig

from corebehrt.common.config import Config, instantiate
from corebehrt.common.loader import ModelLoader, load_model_cfg_from_checkpoint
from corebehrt.common.setup import CHECKPOINTS_DIR
from corebehrt.data.utils import Utilities
from corebehrt.evaluation.utils import get_sampler
from corebehrt.classes.model import BertEHRModel, BertForFineTuning

logger = logging.getLogger(__name__)  # Get the logger for this module


class Initializer:
    """Initialize model, optimizer and scheduler."""

    def __init__(self, cfg: Config, checkpoint: dict = None, model_path: str = None):
        self.cfg = cfg
        self.checkpoint = checkpoint
        if checkpoint:
            self.loader = ModelLoader(cfg, model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    def initialize_pretrain_model(self, train_dataset):
        """Initialize model from checkpoint or from scratch."""
        if self.checkpoint:
            logger.info("Loading model from checkpoint")
            model = self.loader.load_model(BertEHRModel, checkpoint=self.checkpoint)
            model.to(self.device)
        else:
            logger.info("Initializing new model")
            vocab_size = len(train_dataset.vocabulary)
            model = BertEHRModel(
                BertConfig(
                    **self.cfg.model,
                    vocab_size=vocab_size,
                )
            )
        return model

    def initialize_finetune_model(self, train_dataset):
        if self.checkpoint:
            logger.info("Loading model from checkpoint")
            add_config = {**self.cfg.model}
            model = self.loader.load_model(
                BertForFineTuning,
                checkpoint=self.checkpoint,
                add_config=add_config,
            )
            model.to(self.device)
            return model
        else:
            raise NotImplementedError("Fine-tuning from scratch is not supported.")

    def initialize_optimizer(self, model):
        """Initialize optimizer from checkpoint or from scratch."""
        if self.checkpoint:
            logger.info("Loading AdamW optimizer from checkpoint")
            optimizer = AdamW(
                model.parameters(),
            )
            self.optimizer_state_dic_to_device(self.checkpoint["optimizer_state_dict"])
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            return optimizer
        else:
            logger.info("Initializing new AdamW optimizer")
            return AdamW(model.parameters(), **self.cfg.optimizer)

    def initialize_scheduler(self, optimizer):
        """Initialize scheduler from checkpoint or from scratch."""
        if not self.cfg.get("scheduler", None):
            return None
        logger.info("Initializing new scheduler")
        scheduler = instantiate(self.cfg.scheduler, **{"optimizer": optimizer})

        if not self.checkpoint:
            return scheduler

        logger.info("Loading scheduler_state_dict from checkpoint")
        scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])
        return scheduler

    def initialize_sampler(self, train_dataset) -> Tuple[Optional[Sampler], Config]:
        """Initialize sampler and modify cfg."""
        sampler = get_sampler(self.cfg, train_dataset, train_dataset.outcomes)
        if sampler:
            self.cfg.trainer_args.shuffle = False
        return sampler, self.cfg

    def optimizer_state_dic_to_device(self, optimizer_state_dic):
        """Move optimizer state dict to device."""
        for state in optimizer_state_dic["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)


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

        if self.check_checkpoints(self.model_path):
            # Given model has checkpoints -> restart
            self.restart_model_path = self.model_path
        else:
            # Restart model from other directory (if given)
            self.restart_model_path = self.check_model("restart_model", fold=fold)

        # Update config from old model, if relevant
        if self.restart_model_path is not None:
            self.cfg.model = load_model_cfg_from_checkpoint(
                self.restart_model_path, "finetune_config"
            )

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

    def load_checkpoint(self):
        return ModelLoader(self.cfg, self.checkpoint_model_path).load_checkpoint()

    def initialize_finetune_model(self, checkpoint, train_dataset):
        logger.info("Initializing model")
        self.initializer = Initializer(
            self.cfg, checkpoint=checkpoint, model_path=self.checkpoint_model_path
        )
        model = self.initializer.initialize_finetune_model(train_dataset)
        return model

    def initialize_training_components(self, model, train_dataset):
        """Initialize training components. If no model_path provided, optimizer and scheduler are initialized from scratch."""
        if self.restart_model_path is None:
            logger.info("Initializing optimizer and scheduler from scratch")
            self.initializer.checkpoint = None
        optimizer = self.initializer.initialize_optimizer(model)
        sampler, cfg = self.initializer.initialize_sampler(train_dataset)
        scheduler = self.initializer.initialize_scheduler(optimizer)
        return optimizer, sampler, scheduler, cfg

    def get_epoch(self):
        """Get epoch from model_path."""
        if self.restart_model_path is None:
            return 0
        else:
            return Utilities.get_last_checkpoint_epoch(
                join(self.restart_model_path, CHECKPOINTS_DIR)
            )
