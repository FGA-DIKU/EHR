from os.path import join
from typing import Tuple

from corebehrt.functional.setup.model import get_last_checkpoint_epoch
from corebehrt.constants.paths import CHECKPOINTS_DIR
from corebehrt.modules.setup.loader import ModelLoader


def load_checkpoint_and_epoch(model_dir: str, checkpoint_epoch: str = None) -> Tuple:
    """Load checkpoint and epoch from config."""
    checkpoint = ModelLoader(
        model_dir, checkpoint_epoch=checkpoint_epoch
    ).load_checkpoint()
    if checkpoint is not None:
        epoch = checkpoint["epoch"]
    else:
        epoch = get_last_checkpoint_epoch(join(model_dir, CHECKPOINTS_DIR))
    return checkpoint, epoch
