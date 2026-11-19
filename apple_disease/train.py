"""Train one of the paper's three models on the labeled Kaggle images."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import CLASSES
from .data import LeafDataset, load_or_make_split
from .models import MODEL_NAMES, build_model


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str):
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but PyTorch cannot use Apple Metal")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ys, preds, losses = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(criterion(logits, labels).item())
        ys.extend(labels.cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
    return {
        "val_loss": float(np.mean(losses)),
        "val_accuracy": float(accuracy_score(ys, preds)),
        "val_macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/plant-pathology-2020-fgvc7"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=20, help="Paper setting: 20")
    parser.add_argument("--batch-size", type=int, default=128, help="Paper setting: 128")
    parser.add_argument("--image-size", type=int, default=512, help="Paper setting: 512")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Keras Adam default used when paper omits the value")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Small number of training images for a pipeline smoke run")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip weight download for offline smoke runs")
    parser.add_argument("--no-canny", action="store_true",
                        help="Disable the paper-described Canny edge augmentation")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    rows = load_or_make_split(args.data_dir, args.output_dir,
                              args.validation_fraction, args.seed)
    train_rows = rows[rows["split"] == "train"].copy()
    valid_rows = rows[rows["split"] == "valid"].copy()
    if args.limit:
        train_rows = train_rows.sample(min(args.limit, len(train_rows)), random_state=args.seed)

    model, model_config = build_model(args.model, pretrained=not args.no_pretrained)
    model = model.to(device)
    input_mean, input_std = model_config["mean"], model_config["std"]
    train_data = LeafDataset(args.data_dir, train_rows, args.image_size, True,
                             input_mean, input_std, canny=not args.no_canny)
    valid_data = LeafDataset(args.data_dir, valid_rows, args.image_size, False,
                             input_mean, input_std)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=device.type == "cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    best_accuracy = -1.0
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.model}.pt"
    print(f"Model: {args.model} ({MODEL_NAMES[args.model]})")
    print(f"Train images: {len(train_data)} | validation images: {len(valid_data)}")
    print(f"Device: {device} | epochs: {args.epochs} | batch size: {args.batch_size} | image: {args.image_size}px")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start = time.time()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metrics = validate(model, valid_loader, device)
        row = {"epoch": epoch + 1, "train_loss": total_loss / max(1, len(train_loader)),
               **metrics, "seconds": time.time() - start}
        history.append(row)
        print("Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              "val_accuracy={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f}".format(**row))

        # Keep the best validation-accuracy checkpoint for later evaluation.
        if metrics["val_accuracy"] > best_accuracy:
            best_accuracy = metrics["val_accuracy"]
            torch.save({
                "model_key": args.model,
                "model_name": MODEL_NAMES[args.model],
                "classes": CLASSES,
                "image_size": args.image_size,
                "mean": input_mean,
                "std": input_std,
                "state_dict": model.state_dict(),
            }, checkpoint_path)
    pd.DataFrame(history).to_csv(args.output_dir / f"{args.model}_training.csv", index=False)
    with (args.output_dir / f"{args.model}_settings.json").open("w") as f:
        json.dump(vars(args), f, default=str, indent=2)
    print(f"Saved best validation checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
