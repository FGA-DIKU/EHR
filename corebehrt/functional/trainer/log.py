import logging

logger = logging.getLogger(__name__)


def log_number_of_trainable_parameters(model):
    """Prints a minimal summary of which model components are frozen/trainable."""
    # Track parameter status by major component
    embedding_params = 0
    embedding_trainable = 0

    # Collect encoder layer information
    encoder_layers = {}

    head_params = 0
    head_trainable = 0

    total_params = 0
    total_trainable = 0

    # Count parameters by major component
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        is_trainable = param.requires_grad
        if is_trainable:
            total_trainable += num_params

        # Categorize by component
        if name.startswith("embeddings"):
            embedding_params += num_params
            if is_trainable:
                embedding_trainable += num_params
        elif name.startswith("cls"):
            head_params += num_params
            if is_trainable:
                head_trainable += num_params
        elif name.startswith("layers."):
            # Extract layer number
            parts = name.split(".")
            if len(parts) >= 2:
                layer_idx = int(parts[1])
                if layer_idx not in encoder_layers:
                    encoder_layers[layer_idx] = {"trainable": 0, "total": 0}

                encoder_layers[layer_idx]["total"] += num_params
                if is_trainable:
                    encoder_layers[layer_idx]["trainable"] += num_params

    # Print the concise summary
    logger.info("-" * 50)

    # Embedding status
    if embedding_params > 0:
        if embedding_trainable > 0:
            emb_status = f"TRAINABLE ({embedding_trainable/embedding_params*100:.1f}%)"
        else:
            emb_status = "FROZEN"
        logger.info(f"Embeddings: {emb_status}")

    # Encoder layers status
    if encoder_layers:
        num_encoder_layers = len(encoder_layers)
        trainable_layers = sum(
            1 for _, stats in encoder_layers.items() if stats["trainable"] > 0
        )
        logger.info(
            f"Encoder: {trainable_layers}/{num_encoder_layers} layers TRAINABLE"
        )

    # Classification head status
    if head_params > 0:
        if head_trainable > 0:
            head_status = f"TRAINABLE ({head_trainable/head_params*100:.1f}%)"
        else:
            head_status = "FROZEN"
        logger.info(f"Classification head: {head_status}")

    # Overall stats
    logger.info("-" * 50)
    logger.info(
        f"Total: {total_trainable/total_params*100:.1f}% trainable ({total_trainable:,}/{total_params:,} parameters)"
    )
    logger.info("-" * 50 + "\n")
