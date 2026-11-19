"""Evaluate saved models and the paper's mean-probability ensemble."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader

from . import CLASSES
from .data import LeafDataset, load_or_make_split
from .models import MODEL_NAMES, build_model
from .train import choose_device


@torch.no_grad()
def probabilities(model_key, checkpoint_path, data_dir, rows, batch_size, workers, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, config = build_model(model_key, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    dataset = LeafDataset(data_dir, rows, checkpoint["image_size"], False,
                          checkpoint["mean"], checkpoint["std"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    result = []
    for images, _ in loader:
        result.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
    return np.concatenate(result)


def report(name, y_true, probs, output_dir):
    y_pred = probs.argmax(axis=1)
    scores = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    print(f"\n{name}\n{classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)}")
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"{name} confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{name}.png", dpi=160)
    plt.close()
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/plant-pathology-2020-fgvc7"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    split = load_or_make_split(args.data_dir, args.output_dir)
    valid = split[split["split"] == "valid"].copy().reset_index(drop=True)
    y_true = valid["label"].to_numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_probs, rows = [], []
    for key in args.models:
        checkpoint_path = args.output_dir / "checkpoints" / f"{key}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Train {key} first; checkpoint not found: {checkpoint_path}")
        probs = probabilities(key, checkpoint_path, args.data_dir, valid,
                              args.batch_size, args.workers, device)
        all_probs.append(probs)
        rows.append(report(key, y_true, probs, args.output_dir))

    if len(all_probs) > 1:
        ensemble_probs = np.mean(all_probs, axis=0)
        rows.append(report("ensemble", y_true, ensemble_probs, args.output_dir))
    result_path = args.output_dir / "evaluation.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
