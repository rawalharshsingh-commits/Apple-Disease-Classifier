"""Predict a disease class for one uploaded or local leaf image."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

from . import CLASSES
from .models import MODEL_NAMES, build_model
from .train import choose_device


def load_one(model_key, image_path, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _ = build_model(model_key, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = Compose([
        Resize(checkpoint["image_size"], checkpoint["image_size"]),
        Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
        ToTensorV2(),
    ])
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", choices=[*MODEL_NAMES, "ensemble"], default="ensemble")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    model_keys = list(MODEL_NAMES) if args.model == "ensemble" else [args.model]
    predictions = []
    for key in model_keys:
        path = args.output_dir / "checkpoints" / f"{key}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Train {key} first; checkpoint not found: {path}")
        predictions.append(load_one(key, args.image, path, device))
    probs = np.mean(predictions, axis=0)
    print(f"Prediction: {CLASSES[int(probs.argmax())]}\nClass probabilities:")
    for name, probability in sorted(zip(CLASSES, probs), key=lambda pair: -pair[1]):
        print(f"  {name:20} {probability:.4f}")


if __name__ == "__main__":
    main()
