"""Inspect the SmartRupee training dataset without changing image files."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "validation", "test")
CLASSES = ("genuine", "fake")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Dataset root directory")
    args = parser.parse_args()
    root = Path(args.dataset)

    missing = []
    empty = []
    hashes = defaultdict(list)

    print(f"Dataset audit: {root.resolve()}")
    for split in SPLITS:
        for label in CLASSES:
            folder = root / split / label
            if not folder.is_dir():
                missing.append(folder)
                continue
            images = sorted(
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                empty.append(folder)
            print(f"{folder}: {len(images)} images")
            for image in images:
                hashes[sha256(image)].append(image)

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    if duplicate_groups:
        print("\nDuplicate images:")
        for paths in duplicate_groups:
            print("  Hash group:")
            for path in paths:
                print(f"    - {path}")
    else:
        print("\nDuplicate images: none")

    leakage = []
    for paths in duplicate_groups:
        split_locations = {next(path.relative_to(root).parts) for path in paths}
        if len(split_locations) > 1:
            leakage.append((paths, split_locations))

    if leakage:
        print("\nCross-split data leakage detected:")
        for paths, split_locations in leakage:
            print(f"  Splits: {', '.join(sorted(split_locations))}")
            for path in paths:
                print(f"    - {path}")
    else:
        print("Cross-split data leakage: none detected for exact duplicate files")

    if missing:
        print("\nMissing folders:")
        for folder in missing:
            print(f"  - {folder}")
    if empty:
        print("\nEmpty folders (no supported image files):")
        for folder in empty:
            print(f"  - {folder}")

    if not missing and not empty and not duplicate_groups and not leakage:
        print("\nDataset structure is complete and no exact duplicates were found.")


if __name__ == "__main__":
    main()
