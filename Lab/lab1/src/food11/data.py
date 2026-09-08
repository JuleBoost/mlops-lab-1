"""Process the raw Food-11 dataset into resized, category-organized folders.

Reads from ./data/food11_raw/{training,evaluation,validation}, where each image
file is named "<category_index>_<n>.jpg" (category_index 0-10).

Writes two outputs, each split into training/evaluation/validation and then
into one subfolder per category:
  - ./data/food11_processed        (all images, resized to 128x128)
  - ./data/food11_processed_mini   (same, capped at 100 images per category, per split)
"""

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "food11_raw"
PROCESSED_DIR = REPO_ROOT / "data" / "food11_processed"
PROCESSED_MINI_DIR = REPO_ROOT / "data" / "food11_processed_mini"

SPLITS = ["training", "evaluation", "validation"]

CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

TARGET_SIZE = (128, 128)
MINI_MAX_PER_CATEGORY = 100


def process_split(split: str) -> None:
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        print(f"Skipping missing split: {split_dir}")
        return

    mini_counts = {category: 0 for category in CATEGORIES}

    image_paths = sorted(split_dir.glob("*.jpg"))
    for image_path in image_paths:
        try:
            category_index = int(image_path.stem.split("_")[0])
        except (ValueError, IndexError):
            print(f"Skipping unrecognized filename: {image_path.name}")
            continue

        if not (0 <= category_index < len(CATEGORIES)):
            print(f"Skipping out-of-range category index in: {image_path.name}")
            continue

        category = CATEGORIES[category_index]

        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)

            out_dir = PROCESSED_DIR / split / category
            out_dir.mkdir(parents=True, exist_ok=True)
            resized.save(out_dir / image_path.name)

            if mini_counts[category] < MINI_MAX_PER_CATEGORY:
                mini_out_dir = PROCESSED_MINI_DIR / split / category
                mini_out_dir.mkdir(parents=True, exist_ok=True)
                resized.save(mini_out_dir / image_path.name)
                mini_counts[category] += 1

    print(f"Processed split '{split}': {len(image_paths)} images")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MINI_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        process_split(split)

    print("Done.")


if __name__ == "__main__":
    main()
