"""Extract the Kaggle ZIP safely and create the paper's validation split."""
import argparse
import zipfile
from pathlib import Path, PurePosixPath

from .data import make_split, read_labels


def extract_dataset(zip_path: Path, data_dir: Path):
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"train.csv", "test.csv", "sample_submission.csv"}
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            is_image = len(name.parts) == 2 and name.parts[0] == "images" and name.suffix.lower() == ".jpg"
            if not (is_image or (len(name.parts) == 1 and name.name in allowed)):
                continue
            destination = (data_dir / Path(*name.parts)).resolve()
            if data_dir.resolve() not in destination.parents:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
            if member.is_dir():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
            count += 1
    if not (data_dir / "train.csv").exists():
        raise ValueError("The ZIP did not contain train.csv")
    print(f"Extracted {count} files to {data_dir}")
    print(f"Labeled rows in train.csv: {len(read_labels(data_dir))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="Downloaded Kaggle ZIP file")
    parser.add_argument("--data-dir", type=Path, default=Path("data/plant-pathology-2020-fgvc7"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    extract_dataset(args.zip, args.data_dir)
    make_split(args.data_dir, args.output_dir, args.validation_fraction, args.seed)


if __name__ == "__main__":
    main()
