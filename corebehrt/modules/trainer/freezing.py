import logging

import torch

logger = logging.getLogger(__name__)


def freeze_bottom_layers(model: torch.nn.Module, n_layers: int) -> torch.nn.Module:
    """Freezes the bottom n_layers of the transformer model."""
    logger.info(f"Freezing bottom {n_layers} layers of the model")

    # Most transformer models have a 'encoder.layer' or 'layers' attribute
    if hasattr(model, "encoder"):
        layers = model.encoder.layer
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        logger.warning(
            "Warning: Could not detect model layers structure. Freezing skipped."
        )
        return

    # Freeze embeddings if they exist
    if hasattr(model, "embeddings"):
        for param in model.embeddings.parameters():
            param.requires_grad = False

    # Freeze the specified number of layers
    for i in range(n_layers):
        if i < len(layers):
            for param in layers[i].parameters():
                param.requires_grad = False

    return model


def unfreeze_all_layers(model: torch.nn.Module) -> torch.nn.Module:
    """Unfreezes all layers of the model.

    Args:
        model: The model to unfreeze

    Returns:
        The model with all parameters unfrozen
    """
    logger.info("Unfreezing all layers of the model")

    for param in model.parameters():
        param.requires_grad = True

    return model
