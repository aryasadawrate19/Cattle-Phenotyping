"""
Phenotyping Pipeline
Orchestrates the full pipeline: detection → segmentation → feature extraction → trait prediction.
"""

import cv2
import numpy as np

from models.detector_yolov8 import CowDetector
from models.segmenter_sam import CowSegmenter
from models.trait_model_xgboost import TraitPredictor
from pipeline.feature_extractor import FeatureExtractor


class PhenotypingPipeline:
    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        sam_checkpoint: str = "sam_vit_b_01ec64.pth",
        trait_model_dir: str = "saved_models",
    ):
        print("Loading cow detector (YOLOv8n)...")
        self.detector = CowDetector(model_path=yolo_model)

        print("Loading segmenter (SAM ViT-B)...")
        self.segmenter = CowSegmenter(checkpoint_path=sam_checkpoint)

        self.feature_extractor = FeatureExtractor()

        print("Loading trait predictor (XGBoost)...")
        self.predictor = TraitPredictor(model_dir=trait_model_dir)

        print("Pipeline ready.")

    def run(self, image: np.ndarray) -> dict:
        """
        Run the full phenotyping pipeline on an image.

        Args:
            image: BGR numpy array.

        Returns:
            dict with detection info, features, traits, and intermediate outputs.
        """
        result = {}

        # Stage 1 — Cow Detection
        detection = self.detector.detect(image)
        result["cow_detected"] = detection["cow_detected"]
        result["detection_confidence"] = detection["confidence"]
        result["bbox"] = detection["bbox"]

        if not detection["cow_detected"]:
            result["message"] = "No cow detected in the image."
            return result

        bbox = detection["bbox"]

        # Stage 2 — Cow Segmentation
        mask = self.segmenter.segment(image, bbox)
        result["mask"] = mask

        # Stage 3 — Feature Extraction
        features = self.feature_extractor.extract(mask)
        result["features"] = features

        # Stage 4 — Trait Prediction
        traits = self.predictor.predict(features)
        result["estimated_weight_kg"] = traits["estimated_weight_kg"]
        result["body_condition_score"] = traits["body_condition_score"]

        return result
