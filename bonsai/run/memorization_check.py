"""Memorization check: does a pretrained BONSAI model reproduce its training patients?

Prompt the model with the first X events of a patient's record and check whether
greedy generation reproduces the next Y events exactly. Thereafter, it does 
the same for patients, the model never saw, allowing us to ask,
how much more often training patients are reproduced (the excess) measures memorization.

As we are checking whether the model can *exactly* generate the patient trajectory,
we don't really need to append the predictions to the patient sequence and run 
an additional forward pass. For a causal model, generation reproduces the next Y 
events if each of those Y predictions is correct given the true record before it. One
forward pass per patient therefore answers the question for every X and Y.

    python -m bonsai.run.memorization_check --checkpoint best.ckpt \\
        --data-dir <dir with subject_data_train.pt and subject_data_tuning.pt> \\
        --output-dir memorization
"""

import argparse
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bonsai.functional.normalization import normalize_segments
from bonsai.modules.networks.bonsai_nets import (
    BonsaiBase,
    BonsaiPretrain,
    BonsaiValuePretrain,
)

INPUT_FIELDS = ("code", "age", "abspos", "segment", "numeric_value")


def load_model(checkpoint: Path, device: str) -> BonsaiBase:
    """Rebuild the pretrained model (with its next-event head) from a checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = saved["hyper_parameters"]
    if not hparams.get("causal"):
        raise ValueError("The check needs a causally pretrained model (causal: true).")
    cls = BonsaiValuePretrain if hparams.get("value_embedding_mode") else BonsaiPretrain
    accepted = inspect.signature(cls).parameters
    model = cls(
        **{k: v for k, v in hparams.items() if k in accepted}, attention_dropout=0.0
    )
    weights = {
        k.removeprefix("model."): v
        for k, v in saved["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(weights, strict=True)
    return model.eval().to(device)


def load_subjects(data_dir: Path, split: str) -> list[dict]:
    """The processed patients of one split (``subject_data_<split>.pt``)."""
    return torch.load(Path(data_dir) / f"subject_data_{split}.pt", weights_only=False)


def make_batch(subjects: list[dict], length: int, device: str) -> dict:
    """The first ``length`` events of each patient, as the model reads them."""
    batch = {}
    for field in INPUT_FIELDS:
        if all(field in s for s in subjects):
            rows = [s[field][:length] for s in subjects]
            if field == "segment":
                rows = [normalize_segments(row) for row in rows]
            batch[field] = torch.stack(rows).to(device)
    batch["attention_mask"] = torch.ones_like(batch["code"], dtype=torch.bool)
    return batch


@torch.inference_mode()
def correct_next_events(
    model: BonsaiBase, subjects: list[dict], length: int, device: str, batch_size: int
) -> np.ndarray:
    """Boolean [patients, length - 1]: column i says whether event i + 1 was predicted."""
    out = []
    for start in range(0, len(subjects), batch_size):
        batch = make_batch(subjects[start : start + batch_size], length, device)
        with torch.autocast("cuda", torch.float16, enabled=device.startswith("cuda")):
            # Causal attention: the state at position i has seen events 0..i only,
            # so its top prediction is the model's guess for event i + 1.
            states = BonsaiBase.forward(model, batch)
            predicted = model.pretrain_head(states[:, :-1]).argmax(-1)
        out.append((predicted == batch["code"][:, 1:]).cpu().numpy())
    return np.concatenate(out)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (Wilson, 1927).

    Unlike p +/- 1.96 * SE, it stays within [0, 1] and is not zero-width when no
    patient matches, which is the usual case here (Brown, Cai & DasGupta, 2001).
    """
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def sample_patients(subjects: list[dict], min_events: int, n: int) -> list[dict]:
    """Up to ``n`` patients with at least ``min_events`` events (fixed seed)."""
    eligible = [s for s in subjects if len(s["code"]) >= min_events]
    if len(eligible) <= n:
        return eligible
    keep = np.random.default_rng(0).choice(len(eligible), n, replace=False)
    return [eligible[i] for i in sorted(keep)]


def memorization_table(
    model: BonsaiBase,
    subjects_by_split: dict[str, list[dict]],
    prompt_lengths: list[int],
    continuation_lengths: list[int],
    max_patients: int,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Share of patients reproduced exactly (with 95% interval) per split, X and Y."""
    longest = max(continuation_lengths)
    rows = []
    for split, subjects in subjects_by_split.items():
        for x in prompt_lengths:
            # The same patients for every Y: those with at least X + longest Y events.
            patients = sample_patients(subjects, x + longest, max_patients)
            if not patients:
                continue
            correct = correct_next_events(
                model, patients, x + longest, device, batch_size
            )
            for y in continuation_lengths:
                window = correct[
                    :, x - 1 : x - 1 + y
                ]  # predictions of events X+1 ... X+Y
                matches = int(window.all(axis=1).sum())
                lower, upper = wilson_interval(matches, len(patients))
                rows.append(
                    {
                        "split": split,
                        "prompt_events": x,
                        "continuation_events": y,
                        "patients": len(patients),
                        "exact_match": matches / len(patients),
                        "exact_match_lower": lower,
                        "exact_match_upper": upper,
                        # Each prediction given the true record before it.
                        "accuracy_given_true_history": window.mean(),
                    }
                )
    return pd.DataFrame(rows)


def excess_table(
    table: pd.DataFrame, train_split: str, heldout_split: str
) -> pd.DataFrame:
    """Training minus unseen exact-match rate per X and Y.

    The 95% interval for the difference combines the two Wilson intervals
    (Newcombe, 1998, method 10).
    """
    keys = ["prompt_events", "continuation_events"]
    seen = table[table.split == train_split].set_index(keys)
    unseen = table[table.split == heldout_split].set_index(keys)
    t = seen.join(unseen, lsuffix="_train", rsuffix="_unseen", how="inner")
    p1, p2 = t.exact_match_train, t.exact_match_unseen
    diff = p1 - p2
    return pd.DataFrame(
        {
            "exact_match_train": p1,
            "exact_match_unseen": p2,
            "excess": diff,
            "excess_lower": diff
            - np.hypot(p1 - t.exact_match_lower_train, t.exact_match_upper_unseen - p2),
            "excess_upper": diff
            + np.hypot(t.exact_match_upper_train - p1, p2 - t.exact_match_lower_unseen),
        }
    ).reset_index()


def plot_curves(
    table: pd.DataFrame, path: Path, train_split: str, heldout_split: str
) -> None:
    """Exact-match rate against Y: one colour per X, solid = training, dashed = unseen."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    prompts = sorted(table.prompt_events.unique())
    for colour, x in zip(plt.cm.viridis(np.linspace(0.1, 0.85, len(prompts))), prompts):
        for split, style in ((train_split, "-"), (heldout_split, "--")):
            part = table[(table.prompt_events == x) & (table.split == split)]
            label = f"X = {x:,}" if split == train_split else None
            ax.plot(
                part.continuation_events,
                part.exact_match,
                style,
                color=colour,
                marker="o",
                markersize=3,
                label=label,
            )
    ax.plot([], [], "k-", label="Training patients")
    ax.plot([], [], "k--", label="Unseen patients")
    ax.set(
        xlabel="Events to reproduce after the prompt (Y)",
        ylabel="Patients reproduced exactly",
    )
    ax.set_xscale("log", base=2)
    ax.set_title(
        "Exact reproduction of a patient's next Y events after an X-event prompt",
        fontsize=9,
    )
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--heldout-split", default="tuning")
    parser.add_argument(
        "--prompt-events", type=int, nargs="+", default=[16, 64, 256, 1024, 4096]
    )
    parser.add_argument(
        "--continuation-events", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument(
        "--max-patients", type=int, default=2000, help="per split and prompt length"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    context = model.hparams["max_seqlen"]
    longest = max(args.prompt_events) + max(args.continuation_events)
    if longest > context:
        parser.error(
            f"prompt + continuation reaches {longest} events, "
            f"but the model reads at most {context}."
        )
    splits = {
        s: load_subjects(args.data_dir, s)
        for s in (args.train_split, args.heldout_split)
    }
    table = memorization_table(
        model,
        splits,
        args.prompt_events,
        sorted(args.continuation_events),
        args.max_patients,
        args.device,
        args.batch_size,
    )
    excess = excess_table(table, args.train_split, args.heldout_split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "memorization.csv", index=False)
    excess.to_csv(args.output_dir / "excess.csv", index=False)
    plot_curves(
        table,
        args.output_dir / "memorization.png",
        args.train_split,
        args.heldout_split,
    )
    print(excess.to_string(index=False))


if __name__ == "__main__":
    main()
