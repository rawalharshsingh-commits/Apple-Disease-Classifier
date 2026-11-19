#!/usr/bin/env python3
"""Generate a small, deterministic Plant Pathology-shaped fake dataset."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]


def make_dataset(out_dir: Path, count: int = 60, seed: int = 2026) -> None:
    if count < len(CLASSES):
        raise ValueError(f"count must be at least {len(CLASSES)}")

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        image_id = f"fake_{index:04d}"
        label = index % len(CLASSES)

        # Random color backgrounds plus independent texture make each image
        # visibly different and avoid collapsing into duplicate hash groups.
        base = rng.integers(35, 221, size=(1, 1, 3), dtype=np.uint8)
        texture = rng.integers(0, 36, size=(128, 128, 3), dtype=np.uint8)
        image = np.clip(base.astype(np.int16) + texture.astype(np.int16) - 18,
                        0, 255).astype(np.uint8)
        if not cv2.imwrite(str(images_dir / f"{image_id}.jpg"), image):
            raise OSError(f"Could not write image {image_id}")

        row = {"image_id": image_id}
        row.update({name: int(i == label) for i, name in enumerate(CLASSES)})
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image_id", *CLASSES]).to_csv(
        out_dir / "train.csv", index=False)
    print(f"Generated {count} fake images and {out_dir / 'train.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path("tests/fake_data"))
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    make_dataset(args.out_dir, args.count, args.seed)


if __name__ == "__main__":
    main()
