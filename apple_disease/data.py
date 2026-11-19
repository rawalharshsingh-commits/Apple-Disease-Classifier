"""Read the Kaggle CSV, make the paper's holdout split, and load leaf images."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from . import CLASSES


def read_labels(data_dir: Path) -> pd.DataFrame:
    """Read one-hot Kaggle labels and add one integer class label per image."""
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected Kaggle labels at {csv_path}")
    frame = pd.read_csv(csv_path)
    missing = {"image_id", *CLASSES} - set(frame.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    one_hot = frame[CLASSES].to_numpy()
    if not np.isin(one_hot, [0, 1]).all() or not (one_hot.sum(axis=1) == 1).all():
        raise ValueError("Each train.csv row must have exactly one class marked as 1")
    frame["label"] = one_hot.argmax(axis=1).astype(int)
    return frame[["image_id", "label"]]


def make_split(data_dir: Path, output_dir: Path, val_fraction: float = 0.15,
               seed: int = 42) -> pd.DataFrame:
    """Create and save one reproducible, stratified train/validation split."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    labels = read_labels(data_dir)
    train, valid = train_test_split(
        labels, test_size=val_fraction, random_state=seed, shuffle=True,
        stratify=labels["label"])
    split = pd.concat([
        train.assign(split="train"), valid.assign(split="valid")
    ], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "split.csv"
    split.to_csv(path, index=False)
    print(f"Labeled images: {len(labels)}")
    print("All class counts:")
    print(labels["label"].map(dict(enumerate(CLASSES))).value_counts().reindex(CLASSES))
    print("\nTrain counts:")
    print(split[split["split"] == "train"]["label"].map(dict(enumerate(CLASSES)).get).value_counts().reindex(CLASSES))
    print("\nValidation counts:")
    print(split[split["split"] == "valid"]["label"].map(dict(enumerate(CLASSES)).get).value_counts().reindex(CLASSES))
    print(f"\nSaved split to {path}")
    return split


def load_or_make_split(data_dir: Path, output_dir: Path,
                       val_fraction: float = 0.15, seed: int = 42) -> pd.DataFrame:
    path = output_dir / "split.csv"
    if path.exists():
        split = pd.read_csv(path)
        expected = set(read_labels(data_dir)["image_id"])
        if set(split["image_id"]) == expected and set(split["split"]) == {"train", "valid"}:
            return split
    return make_split(data_dir, output_dir, val_fraction, seed)


def make_transforms(image_size: int, training: bool, mean, std):
    """Apply online augmentation to training only; validation stays deterministic."""
    steps = [A.Resize(image_size, image_size)]
    if training:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.25),
        ]
    steps += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(steps)


class LeafDataset(Dataset):
    """Load one image at a time and augment it only when training."""
    def __init__(self, data_dir: Path, rows: pd.DataFrame, image_size: int,
                 training: bool, mean, std, canny: bool = False):
        self.images_dir = data_dir / "images"
        self.rows = rows.reset_index(drop=True)
        self.transform = make_transforms(image_size, training, mean, std)
        self.canny = canny and training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        path = self.images_dir / f"{row.image_id}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.canny and np.random.random() < 0.25:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            image = np.repeat(edges[:, :, None], 3, axis=2)
        image = self.transform(image=image)["image"]
        return image, int(row.label)
