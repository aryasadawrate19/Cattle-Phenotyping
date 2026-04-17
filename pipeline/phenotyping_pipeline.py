"""
Phenotyping Pipeline
Orchestrates the full pipeline: detection → segmentation → feature extraction → trait prediction.
"""

import os
import cv2
import numpy as np

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


class PhenotypingPipeline:
    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        sam_checkpoint: str = "sam_vit_b_01ec64.pth",
        trait_model_dir: str = "saved_models",
    ):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        yolo_model_path = yolo_model if os.path.isabs(yolo_model) else os.path.join(project_root, yolo_model)
        sam_checkpoint_path = sam_checkpoint if os.path.isabs(sam_checkpoint) else os.path.join(project_root, sam_checkpoint)
        trait_model_path = trait_model_dir if os.path.isabs(trait_model_dir) else os.path.join(project_root, trait_model_dir)

        print("Loading cow detector (YOLOv8n)...")
        self.detector = CowDetector(model_path=yolo_model_path)

        print("Loading segmenter (SAM ViT-B)...")
        self.segmenter = CowSegmenter(checkpoint_path=sam_checkpoint_path)

        self.feature_extractor = FeatureExtractor()

        print("Loading trait predictor (XGBoost)...")
        self.predictor = TraitPredictor(model_dir=trait_model_path)

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

        # ------------------------------------------------------------------
        # STEP 0 — Standardize Image Resolution
        # ------------------------------------------------------------------
        image = resize_image_keep_aspect(image)

        result["processed_image"] = image

        # ------------------------------------------------------------------
        # Stage 1 — Cow Detection
        # ------------------------------------------------------------------
        detection = self.detector.detect(image)

        result["cow_detected"] = detection["cow_detected"]
        result["detection_confidence"] = detection["confidence"]
        result["bbox"] = detection["bbox"]

        if not detection["cow_detected"]:
            result["message"] = "No cow detected in the image."
            return result

        bbox = detection["bbox"]

        # ------------------------------------------------------------------
        # Stage 2 — Cow Segmentation
        # ------------------------------------------------------------------
        mask = self.segmenter.segment(image, bbox)

        result["mask"] = mask

        # ------------------------------------------------------------------
        # Stage 3 — Feature Extraction
        # ------------------------------------------------------------------
        features = self.feature_extractor.extract(mask)

        result["features"] = features

        # ------------------------------------------------------------------
        # Stage 4 — Trait Prediction
        # ------------------------------------------------------------------
        traits = self.predictor.predict(features)

        result["estimated_weight_kg"] = traits["estimated_weight_kg"]
        result["body_condition_score"] = traits["body_condition_score"]
        result["body_length_cm"] = traits.get("body_length_cm", 0.0)
        result["withers_height_cm"] = traits.get("withers_height_cm", 0.0)
        result["heart_girth_cm"] = traits.get("heart_girth_cm", 0.0)
        result["hip_length_cm"] = traits.get("hip_length_cm", 0.0)

        return result