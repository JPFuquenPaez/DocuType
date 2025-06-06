"""This script processes the "CATMuS/modern" dataset (https://huggingface.co/datasets/CATMuS/), which contains images of printed and handwritten text.

It saves the images into separate folders based on the split (train, validation, test) and writing type (handwritten, printed).
"""

import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_BASE = Path("datasets/printed-handwritten-text-images")
SPLITS = ["train", "validation", "test"]


def process_split(split: str):
    """Processes a dataset split by loading images and saving them into subfolders based on writing type."""
    logger.info(f"Processing {split} split")
    try:
        dataset = load_dataset("CATMuS/modern", split=split)
    except Exception as e:
        logger.error(f"Failed to load {split}: {e}")
        return

    for idx, example in enumerate(dataset):
        try:
            img = example["im"].convert("RGB")
            writing_type = example.get(
                "writing_type", "unknown"
            )  # will be "handwritten" or "printed"

            #  subfolder based on type
            output_dir = OUTPUT_BASE / split / writing_type
            output_dir.mkdir(parents=True, exist_ok=True)

            img.save(output_dir / f"{split}_{idx:05d}_{writing_type}.png")
        except Exception as e:
            logger.error(f"Error processing {split} sample {idx}: {str(e)[:50]}")

    logger.info(f"Finished processing {split} split with {len(dataset)} samples")


def main():
    """Iterates over predefined data splits and processes each split."""
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
