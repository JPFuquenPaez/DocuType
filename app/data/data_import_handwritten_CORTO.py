"""Module for extracting and saving images from corto-ai handwritten text dataset.

This module provides functionality to read image data stored in Parquet format,
extract the images, and save them as PNG files. It is designed to process datasets
split into training, validation, and testing sets, and organizes the output images
into corresponding directories.
"""

import io
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


SPLITS = {
    "train": "hf://datasets/corto-ai/handwritten-text/data/train-00000-of-00001.parquet",
    "valid": "hf://datasets/corto-ai/handwritten-text/data/valid-00000-of-00001.parquet",
    "test": "hf://datasets/corto-ai/handwritten-text/data/test-00000-of-00001.parquet",
}
OUTPUT_BASE = Path("datasets/handwritten-text-images")


def ensure_dir(path: Path):
    """Ensures that the directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ready: {path}")


def dump_images_only(parquet_uri: str, split: str, out_base: Path):
    """Extracts images from a Parquet file and saves them as PNG."""
    logger.info(f"Reading {split} from {parquet_uri}")
    df = pd.read_parquet(parquet_uri)
    logger.info(f" → {len(df)} samples")

    split_dir = out_base / split
    ensure_dir(split_dir)

    for idx, row in df.iterrows():
        img_bytes = row["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("L")

        # single class 0
        filename = f"{split}_{idx:05d}_0.png"
        img.save(split_dir / filename)

    logger.info(f"Saved {len(df)} images to {split_dir}")


def main():
    """Main function to process image data for each dataset split.

    Raises:
        Exception: If an error occurs during the processing of any split.
    """
    for split, uri in SPLITS.items():
        try:
            dump_images_only(uri, split, OUTPUT_BASE)
        except Exception as e:
            logger.error(f"Error in {split}: {e}")
            raise


if __name__ == "__main__":
    main()
