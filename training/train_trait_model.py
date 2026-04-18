"""
Train Trait Model
Extracts morphological features from labelled images and trains XGBoost regressors
for weight and BCS prediction.

Usage:
    python training/train_trait_model.py --data_dir data/images --labels data/labels.csv
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

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

    morphometric_features = [
        "body_length_cm",
        "withers_height_cm",
        "heart_girth_cm",
        "hip_length_cm",
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

        # Append morphometric measurements from labels.csv
        for morph_feat in morphometric_features:
            vec.append(row[morph_feat])

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

    # =========================================================================
    # 5-FOLD CROSS-VALIDATION
    # =========================================================================
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {
        "weight_mae_scores": [],
        "weight_r2_scores": [],
        "bcs_mae_scores": [],
        "bcs_r2_scores": [],
        "folds": []
    }
    
    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 80 + "\n")
    
    fold_num = 0
    for train_idx, val_idx in kfold.split(X):
        fold_num += 1
        X_train, X_val = X[train_idx], X[val_idx]
        y_weight_train, y_weight_val = y_weight[train_idx], y_weight[val_idx]
        y_bcs_train, y_bcs_val = y_bcs[train_idx], y_bcs[val_idx]
        
        print(f"Fold {fold_num}/5: Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        # Train weight model for this fold
        weight_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        weight_model.fit(X_train, y_weight_train)
        
        # Evaluate weight model
        y_weight_pred = weight_model.predict(X_val)
        weight_mae = mean_absolute_error(y_weight_val, y_weight_pred)
        weight_r2 = r2_score(y_weight_val, y_weight_pred)
        
        cv_results["weight_mae_scores"].append(float(weight_mae))
        cv_results["weight_r2_scores"].append(float(weight_r2))
        
        # Train BCS model for this fold
        bcs_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        bcs_model.fit(X_train, y_bcs_train)
        
        # Evaluate BCS model
        y_bcs_pred = bcs_model.predict(X_val)
        bcs_mae = mean_absolute_error(y_bcs_val, y_bcs_pred)
        bcs_r2 = r2_score(y_bcs_val, y_bcs_pred)
        
        cv_results["bcs_mae_scores"].append(float(bcs_mae))
        cv_results["bcs_r2_scores"].append(float(bcs_r2))
        
        fold_result = {
            "fold": fold_num,
            "weight_mae": float(weight_mae),
            "weight_r2": float(weight_r2),
            "bcs_mae": float(bcs_mae),
            "bcs_r2": float(bcs_r2)
        }
        cv_results["folds"].append(fold_result)
        
        print(f"  Weight - MAE: {weight_mae:.4f}, R²: {weight_r2:.4f}")
        print(f"  BCS    - MAE: {bcs_mae:.4f}, R²: {bcs_r2:.4f}\n")
    
    # =========================================================================
    # CROSS-VALIDATION SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<15} {'Weight':<20} {'BCS':<20}")
    print("-" * 55)
    
    weight_mae_mean = np.mean(cv_results["weight_mae_scores"])
    weight_mae_std = np.std(cv_results["weight_mae_scores"])
    weight_r2_mean = np.mean(cv_results["weight_r2_scores"])
    weight_r2_std = np.std(cv_results["weight_r2_scores"])
    
    bcs_mae_mean = np.mean(cv_results["bcs_mae_scores"])
    bcs_mae_std = np.std(cv_results["bcs_mae_scores"])
    bcs_r2_mean = np.mean(cv_results["bcs_r2_scores"])
    bcs_r2_std = np.std(cv_results["bcs_r2_scores"])
    
    print(f"{'MAE':<15} {weight_mae_mean:.4f} ± {weight_mae_std:.4f}    {bcs_mae_mean:.4f} ± {bcs_mae_std:.4f}")
    print(f"{'R² Score':<15} {weight_r2_mean:.4f} ± {weight_r2_std:.4f}    {bcs_r2_mean:.4f} ± {bcs_r2_std:.4f}")
    print("\n" + "=" * 80 + "\n")
    
    # Store summary statistics
    cv_results["summary"] = {
        "weight_mae_mean": float(weight_mae_mean),
        "weight_mae_std": float(weight_mae_std),
        "weight_r2_mean": float(weight_r2_mean),
        "weight_r2_std": float(weight_r2_std),
        "bcs_mae_mean": float(bcs_mae_mean),
        "bcs_mae_std": float(bcs_mae_std),
        "bcs_r2_mean": float(bcs_r2_mean),
        "bcs_r2_std": float(bcs_r2_std),
    }
    
    # =========================================================================
    # TRAIN FINAL MODEL ON FULL DATASET
    # =========================================================================
    print("Training final models on full dataset...")
    
    predictor = TraitPredictor(model_dir=args.model_dir)
    predictor.train(X, y_weight, y_bcs)
    
    # =========================================================================
    # SAVE CV RESULTS
    # =========================================================================
    cv_results_path = os.path.join(args.model_dir, "cv_results.json")
    os.makedirs(args.model_dir, exist_ok=True)
    
    with open(cv_results_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"CV results saved to {cv_results_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()