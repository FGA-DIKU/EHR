import logging
from typing import List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import Sampler
from transformers import ModernBertConfig


from corebehrt.modules.setup.config import Config, instantiate_class
from corebehrt.modules.model.model import (
    CorebehrtForPretraining,
    CorebehrtForFineTuning,
)
from corebehrt.modules.setup.loader import ModelLoader
from corebehrt.modules.trainer.utils import get_sampler

logger = logging.getLogger(__name__)


class Initializer:
    """Initialize model, optimizer and scheduler."""

    def __init__(self, cfg: Config, checkpoint: dict = None, model_path: str = None):
        self.cfg = cfg
        self.checkpoint = checkpoint
        if checkpoint:
            self.loader = ModelLoader(model_path, cfg.paths.get("checkpoint_epoch"))
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    def initialize_pretrain_model(self, train_dataset):
        """Initialize model from checkpoint or from scratch."""
        if self.checkpoint:
            logger.info("Loading model from checkpoint")
            model = self.loader.load_model(
                CorebehrtForPretraining, checkpoint=self.checkpoint
            )
            model.to(self.device)
        else:
            logger.info("Initializing new model")
            vocab_size = len(train_dataset.vocabulary)
            model = CorebehrtForPretraining(
                ModernBertConfig(
                    **self.cfg.model,
                    vocab_size=vocab_size,
                    pad_token_id=0,
                    cls_token_id=1,
                    sep_token_id=2,
                    sparse_prediction=True,
                )
            )
        return model

    def initialize_finetune_model(self):
        if self.checkpoint:
            logger.info("Loading model from checkpoint")
            add_config = {**self.cfg.model}
            model = self.loader.load_model(
                CorebehrtForFineTuning,
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
        scheduler = instantiate_class(self.cfg.scheduler, **{"optimizer": optimizer})

        if not self.checkpoint:
            return scheduler

        logger.info("Loading scheduler_state_dict from checkpoint")
        scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])
        return scheduler

    def initialize_sampler(
        self, outcomes: List[int]
    ) -> Tuple[Optional[Sampler], Config]:
        """Initialize sampler and modify cfg."""
        sampler = get_sampler(self.cfg, outcomes)
        if sampler:
            self.cfg.trainer_args.shuffle = False
        return sampler, self.cfg

    def optimizer_state_dic_to_device(self, optimizer_state_dic):
        """Move optimizer state dict to device."""
        for state in optimizer_state_dic["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
