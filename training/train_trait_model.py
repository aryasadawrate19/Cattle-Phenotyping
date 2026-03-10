"""
Train Trait Model
Extracts morphological features from labelled images and trains XGBoost regressors
for weight and BCS prediction.

Usage:
    python training/train_trait_model.py --data_dir data/images --labels data/labels.csv
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.detector_yolov8 import CowDetector
from models.segmenter_sam import CowSegmenter
from models.trait_model_xgboost import TraitPredictor
from pipeline.feature_extractor import FeatureExtractor


TARGET_WIDTH = 1024


def resize_image_keep_aspect(image, target_width=TARGET_WIDTH):
    """
    Resize image to a fixed width while keeping aspect ratio.
    Ensures consistent scale between training and inference.
    """
    h, w = image.shape[:2]

    if w == target_width:
        return image

    scale = target_width / w
    new_h = int(h * scale)

    resized = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)

    return resized


def extract_features_from_images(
    data_dir: str,
    labels_df: pd.DataFrame,
    detector: CowDetector,
    segmenter: CowSegmenter,
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for every labelled image."""

    feature_names = [
        "body_area_px",
        "body_length_px",
        "body_height_px",
        "bbox_width",
        "bbox_height",
        "contour_perimeter",
        "convex_hull_area",
        "aspect_ratio",
        "body_area_ratio",
        "solidity",
        "compactness",
    ]

    X_list, y_weight, y_bcs = [], [], []

    for _, row in labels_df.iterrows():
        img_path = os.path.join(data_dir, row["image_name"])

        if not os.path.exists(img_path):
            print(f"  Skipping {row['image_name']} (file not found)")
            continue

        image = cv2.imread(img_path)

        if image is None:
            print(f"  Skipping {row['image_name']} (unreadable)")
            continue

        # ------------------------------------------------------------------
        # STANDARDIZE IMAGE RESOLUTION (same as inference pipeline)
        # ------------------------------------------------------------------
        image = resize_image_keep_aspect(image)

        detection = detector.detect(image)

        if not detection["cow_detected"]:
            print(f"  Skipping {row['image_name']} (no cow detected)")
            continue

        mask = segmenter.segment(image, detection["bbox"])

        features = extractor.extract(mask)

        vec = [features[name] for name in feature_names]

        X_list.append(vec)
        y_weight.append(row["weight"])
        y_bcs.append(row["bcs"])

        print(f"  Processed {row['image_name']}")

    return np.array(X_list), np.array(y_weight), np.array(y_bcs)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost trait models")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images",
        help="Directory containing cow images",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="data/labels.csv",
        help="CSV file with columns: image_name, weight, bcs",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models",
        help="Directory to save trained models",
    )

    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"Labels file not found: {args.labels}")
        sys.exit(1)

    labels_df = pd.read_csv(args.labels)

    print(f"Loaded {len(labels_df)} labelled samples from {args.labels}")

    detector = CowDetector()
    segmenter = CowSegmenter()
    extractor = FeatureExtractor()

    print("Extracting features from images...")

    X, y_weight, y_bcs = extract_features_from_images(
        args.data_dir,
        labels_df,
        detector,
        segmenter,
        extractor,
    )

    if len(X) == 0:
        print("No valid samples found. Cannot train models.")
        sys.exit(1)

    print(f"Training on {len(X)} samples...")

    predictor = TraitPredictor(model_dir=args.model_dir)

    predictor.train(X, y_weight, y_bcs)

    print("Training complete.")


if __name__ == "__main__":
    main()