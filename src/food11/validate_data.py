"""Check a processed Food-11 dataset against its data contract.

Usage:
    uv run python src/food11/validate_data.py data/food11_processed_mini
"""

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

CLASSES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
           "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
SPLITS = ["training", "validation", "evaluation"]
SIZE = (128, 128)


def images(split_dir: Path, cls: str) -> list[Path]:
    return sorted(p for p in (split_dir / cls).iterdir() if p.is_file())


def check_class_folders(root: Path) -> list[str]:
    errors = []
    for split in SPLITS:
        found = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
        if found != sorted(CLASSES):
            errors.append(f"{split}: expected the 11 classes, found {found}")
    return errors


def check_images(root: Path) -> list[str]:
    errors = []
    for split in SPLITS:
        for cls in CLASSES:
            for path in images(root / split, cls):
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB" or img.size != SIZE:
                            errors.append(f"{path}: {img.mode} {img.size}")
                except (OSError, ValueError) as exc:  # PIL: unreadable / truncated file
                    errors.append(f"{path}: cannot open ({exc})")
    return errors


def check_min_per_class(root: Path, minimum: int) -> list[str]:
    errors = []
    for split in SPLITS:
        for cls in CLASSES:
            n = len(images(root / split, cls))
            if n < minimum:
                errors.append(f"{split}/{cls}: {n} images < {minimum}")
    return errors


def check_no_leakage(root: Path) -> list[str]:
    seen: dict[str, Path] = {}
    errors = []
    for split in SPLITS:
        for cls in CLASSES:
            for path in images(root / split, cls):
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                other = seen.get(digest)
                if other is not None and other.parts[-3] != split:
                    errors.append(f"{path} duplicates {other}")
                seen.setdefault(digest, path)
    return errors


def check_class_balance(root: Path, tau: float) -> list[str]:
    def shares(split: str) -> dict[str, float]:
        counts = {cls: len(images(root / split, cls)) for cls in CLASSES}
        total = sum(counts.values())
        return {cls: counts[cls] / total for cls in CLASSES}

    train, val = shares("training"), shares("validation")
    return [
        f"{cls}: training share {train[cls]:.3f} vs validation {val[cls]:.3f}"
        for cls in CLASSES
        if abs(train[cls] - val[cls]) > tau
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a processed Food-11 dataset")
    p.add_argument("dataset", type=Path)
    p.add_argument("--min-per-class", type=int, default=50)
    p.add_argument("--tau", type=float, default=0.05)
    args = p.parse_args()

    checks = [
        ("11 class folders in every split", lambda: check_class_folders(args.dataset)),
        ("every image is RGB 128x128", lambda: check_images(args.dataset)),
        (f"at least {args.min_per_class} images per class", lambda: check_min_per_class(args.dataset, args.min_per_class)),
        ("no image in two splits", lambda: check_no_leakage(args.dataset)),
        (f"class shares within {args.tau} between training and validation", lambda: check_class_balance(args.dataset, args.tau)),
    ]

    failed = False
    for name, run in checks:
        errors = run()
        print(f"{'PASS' if not errors else 'FAIL'}  {name}")
        for err in errors[:5]:
            print(f"      {err}")
        if len(errors) > 5:
            print(f"      ... and {len(errors) - 5} more")
        failed = failed or bool(errors)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
