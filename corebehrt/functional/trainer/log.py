"""
This provides logging functionality for the model.
Inspired by:
https://lightning.ai/docs/pytorch/1.6.2/_modules/pytorch_lightning/utilities/model_summary.html#ModelSummary
"""

import logging
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)

PARAMETER_NUM_UNITS = ["", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """Formats parameter counts with K, M, B, T suffixes.

    Args:
        number: Number to format

    Returns:
        Formatted string like '1.2K' or '34.5M'
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d}{labels[index]}"
    return f"{number:,.1f}{labels[index]}"


def format_summary_table(
    headers: List[str], rows: List[List[str]], footer_rows: List[Tuple[str, str]]
) -> str:
    """Creates a formatted table string with headers, rows and footer.

    Args:
        headers: Column header strings
        rows: Table row data
        footer_rows: Footer rows as (label, value) pairs

    Returns:
        Formatted table string
    """
    # Get column widths
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))]

    # Format string for each column
    row_format = " | ".join(f"{{:<{width}}}" for width in col_widths)

    # Create table string
    lines = [
        row_format.format(*headers),
        "-" * (sum(col_widths) + 3 * (len(col_widths) - 1)),
    ]
    lines.extend(row_format.format(*row) for row in rows)
    lines.append("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))

    # Add footer
    footer_width = max(len(label) for label, _ in footer_rows)
    footer_format = f"{{:<{footer_width}}} {{}}"
    lines.extend(footer_format.format(label, value) for label, value in footer_rows)

    return "\n".join(lines)


def log_number_of_trainable_parameters(model: nn.Module) -> None:
    """Enhanced parameter logging with detailed breakdown and formatted table.

    Args:
        model: The model to analyze
    """
    # Track parameters by component
    components = OrderedDict()
    total_params = 0
    total_trainable = 0

    # Collect parameter stats
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        is_trainable = param.requires_grad
        if is_trainable:
            total_trainable += num_params

        # Categorize by component
        if name.startswith("embeddings"):
            component = "embeddings"
        elif name.startswith("cls"):
            component = "classification_head"
        elif name.startswith("layers."):
            layer_idx = int(name.split(".")[1])
            component = f"encoder_layer_{layer_idx}"
        else:
            component = "other"

        if component not in components:
            components[component] = {"trainable": 0, "total": 0}
        components[component]["total"] += num_params
        if is_trainable:
            components[component]["trainable"] += num_params

    # Prepare table data
    headers = ["Component", "Parameters", "Trainable", "Status"]
    rows = []

    for name, stats in components.items():
        trainable_pct = (
            stats["trainable"] / stats["total"] * 100 if stats["total"] > 0 else 0
        )
        status = "TRAINABLE" if stats["trainable"] > 0 else "FROZEN"
        rows.append(
            [
                name,
                get_human_readable_count(stats["total"]),
                f"{trainable_pct:.1f}%",
                status,
            ]
        )

    # Prepare footer statistics
    footer_rows = [
        (
            "Trainable params:",
            f"{get_human_readable_count(total_trainable)} ({total_trainable/total_params*100:.1f}%)",
        ),
        (
            "Non-trainable params:",
            get_human_readable_count(total_params - total_trainable),
        ),
        ("Total params:", get_human_readable_count(total_params)),
        (
            "Model size (MB):",
            f"{(total_params * 4) / (1024*1024):.2f}",
        ),  # Assuming float32
    ]

    # Log the formatted table
    logger.info("\nModel Parameter Summary:\n")
    logger.info(format_summary_table(headers, rows, footer_rows))
    logger.info("\n")
