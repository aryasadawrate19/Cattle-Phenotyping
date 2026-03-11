"""
XGBoost Trait Prediction Model
Predicts cattle weight (kg) and body condition score (BCS) from morphological features.
"""

import os
import numpy as np
import xgboost as xgb
import joblib


class TraitPredictor:
    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        self.weight_model = None
        self.bcs_model = None
        self.feature_names = [
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
        self._load_models()

    def _load_models(self):
        """Load saved models if they exist."""
        weight_path = os.path.join(self.model_dir, "weight_model.json")
        bcs_path = os.path.join(self.model_dir, "bcs_model.json")

        if os.path.exists(weight_path):
            self.weight_model = xgb.XGBRegressor()
            self.weight_model.load_model(weight_path)

        if os.path.exists(bcs_path):
            self.bcs_model = xgb.XGBRegressor()
            self.bcs_model.load_model(bcs_path)

    def predict(self, features: dict) -> dict:
        """
        Predict weight and BCS from morphological features.

        Args:
            features: dict with morphological measurements.

        Returns:
            dict with estimated_weight_kg and body_condition_score.
        """
        feature_vector = np.array(
            [[features.get(name, 0) for name in self.feature_names]]
        )

        result = {}

        if self.weight_model is not None:
            result["estimated_weight_kg"] = round(
                float(self.weight_model.predict(feature_vector)[0]), 1
            )
        else:
            # Fallback heuristic when no trained model is available
            result["estimated_weight_kg"] = self._heuristic_weight(features)

        if self.bcs_model is not None:
            raw_bcs = float(self.bcs_model.predict(feature_vector)[0])
            result["body_condition_score"] = round(max(1.0, min(5.0, raw_bcs)), 1)
        else:
            result["body_condition_score"] = self._heuristic_bcs(features)

        return result

    @staticmethod
    def _heuristic_weight(features: dict) -> float:
        """Simple heuristic weight estimate based on pixel area."""
        area = features.get("body_area_px", 0)
        if area == 0:
            return 0.0
        # Rough linear mapping: ~3 px² per kg (tuned for typical side-view photos)
        estimated = area * 0.003 + 50
        return round(max(100.0, min(900.0, estimated)), 1)

    @staticmethod
    def _heuristic_bcs(features: dict) -> float:
        """Simple heuristic BCS estimate based on aspect ratio and area fill."""
        aspect = features.get("aspect_ratio", 1.0)
        area = features.get("body_area_px", 0)
        hull = features.get("convex_hull_area", 1)
        fill_ratio = area / hull if hull > 0 else 0.5

        # Wider and more filled => higher BCS
        bcs = 2.0 + fill_ratio * 2.0 + (1.0 / max(aspect, 0.5)) * 0.5
        return round(max(1.0, min(5.0, bcs)), 1)

    def train(self, X: np.ndarray, y_weight: np.ndarray, y_bcs: np.ndarray):
        """
        Train weight and BCS models.

        Args:
            X: Feature matrix (n_samples, n_features).
            y_weight: Weight targets.
            y_bcs: BCS targets.
        """
        os.makedirs(self.model_dir, exist_ok=True)

        self.weight_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.weight_model.fit(X, y_weight)
        self.weight_model.save_model(
            os.path.join(self.model_dir, "weight_model.json")
        )

        self.bcs_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.bcs_model.fit(X, y_bcs)
        self.bcs_model.save_model(
            os.path.join(self.model_dir, "bcs_model.json")
        )

        print("Models trained and saved.")
