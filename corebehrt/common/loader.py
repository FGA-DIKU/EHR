import logging
import os
from os.path import join
from typing import Tuple

import torch
from transformers import BertConfig

from corebehrt.common.config import Config, load_config
from corebehrt.classes.dataset import PatientDataset
from corebehrt.data.utils import Utilities

logger = logging.getLogger(__name__)  # Get the logger for this module

VOCABULARY_FILE = "vocabulary.pt"
CHECKPOINT_FOLDER = "checkpoints"


def load_checkpoint_and_epoch(model_dir: str, checkpoint_epoch: str = None) -> Tuple:
    """Load checkpoint and epoch from config."""
    checkpoint = ModelLoader(
        model_dir, checkpoint_epoch=checkpoint_epoch
    ).load_checkpoint()
    if checkpoint is not None:
        epoch = checkpoint["epoch"]
    else:
        epoch = Utilities.get_last_checkpoint_epoch(join(model_dir, CHECKPOINT_FOLDER))
    return checkpoint, epoch


def load_model_cfg_from_checkpoint(model_dir: str, config_name: str) -> Config:
    """
    Load the model sub section from the configuration file given in the
    model dir.
    """
    return load_config(join(model_dir, f"{config_name}.yaml")).model


class FeaturesLoader:
    def __init__(self, cfg):
        self.paths_cfg = cfg.paths
        self.cfg = cfg

    def load_vocabulary(self):
        """Load vocabulary from file."""
        return torch.load(join(self.paths_cfg.tokenized, VOCABULARY_FILE))


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
        config = BertConfig.from_pretrained(self.model_path)
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
        return torch.load(checkpoint_path, map_location=device)

    def get_checkpoint_epoch(self) -> int:
        """Get checkpoint if set or return the last checkpoint_epoch for this model."""
        if self.checkpoint_epoch is None:
            logger.info("No checkpoint provided. Loading last checkpoint.")
            self.checkpoint_epoch = Utilities.get_last_checkpoint_epoch(
                join(self.model_path, CHECKPOINT_FOLDER)
            )
        return self.checkpoint_epoch


def get_pids_file(split_dir: str, mode: str) -> str:
    """Get pids file from predefined splits.
    The file can be named pids_{mode}.pt or {mode}_pids.pt."""
    if os.path.exists(join(split_dir, f"{mode}_pids.pt")):
        return join(split_dir, f"{mode}_pids.pt")
    elif os.path.exists(join(split_dir, f"pids_{mode}.pt")):
        return join(split_dir, f"pids_{mode}.pt")
    else:
        raise ValueError(f"No pids file found for mode {mode} in {split_dir}")


def load_and_select_splits(
    split_dir: str, data: PatientDataset
) -> Tuple[PatientDataset, PatientDataset]:
    """Load and select pids from predefined splits"""
    logger.info("Load and select pids")
    train_pids = torch.load(get_pids_file(split_dir, "train"))
    val_pids = torch.load(get_pids_file(split_dir, "val"))
    train_data = data.filter_by_pids(train_pids)
    val_data = data.filter_by_pids(val_pids)
    return train_data, val_data
