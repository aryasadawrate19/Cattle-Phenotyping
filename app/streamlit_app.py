"""
Livestock Phenotyping MVP — Streamlit App
Upload a cow image and get estimated traits (weight, BCS) via the full pipeline.
"""

import sys
import os
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.phenotyping_pipeline import PhenotypingPipeline
from utils.visualization import draw_bbox, overlay_mask, create_mask_visualization

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Livestock Phenotyping MVP", layout="wide")
st.title("🐄 Livestock Phenotyping System")
st.markdown(
    "Upload a **side-view image of a cow** to estimate body condition score (BCS) and weight."
)


# ── Load pipeline (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models (YOLOv8 + SAM + XGBoost)...")
def load_pipeline():
    return PhenotypingPipeline()


pipeline = load_pipeline()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    show_mask_overlay = st.checkbox("Show mask overlay", value=True)
    show_raw_mask = st.checkbox("Show raw segmentation mask", value=False)


# ── Image upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a cow image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Uploaded Image")
    st.image(image_rgb, use_container_width=True)

    # Run pipeline
    with st.spinner("Running phenotyping pipeline..."):
        result = pipeline.run(image_bgr)

    if not result["cow_detected"]:
        st.error("No cow detected in the image. Please upload a clear cattle photo.")
    else:
        col1, col2 = st.columns(2)

        # ── Detection ────────────────────────────────────────────────────
        with col1:
            st.subheader("Detected Cow")
            bbox_vis = draw_bbox(result["processed_image"], result["bbox"])
            st.image(
                cv2.cvtColor(bbox_vis, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        # ── Segmentation ─────────────────────────────────────────────────
        with col2:
            st.subheader("Segmentation Mask")
            if show_mask_overlay:
                mask_vis = overlay_mask(result["processed_image"], result["mask"])
                st.image(
                    cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
            if show_raw_mask:
                raw_mask_vis = create_mask_visualization(result["mask"])
                st.image(raw_mask_vis, use_container_width=True)

        st.divider()

        # ── Features ─────────────────────────────────────────────────────
        st.subheader("Extracted Morphological Features")
        features = result["features"]
        feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
        feat_col1.metric("Body Area (px)", f"{features['body_area_px']:,}")
        feat_col2.metric("Body Length (px)", features["body_length_px"])
        feat_col3.metric("Body Height (px)", features["body_height_px"])
        feat_col4.metric("Aspect Ratio", features["aspect_ratio"])

        feat_col5, feat_col6, feat_col7, _ = st.columns(4)
        feat_col5.metric("Perimeter (px)", features["contour_perimeter"])
        feat_col6.metric("Convex Hull Area (px)", f"{features['convex_hull_area']:,}")
        feat_col7.metric("BBox W × H", f"{features['bbox_width']} × {features['bbox_height']}")

        st.divider()

        # ── Predicted Traits ─────────────────────────────────────────────
        st.subheader("Predicted Livestock Traits")
        trait_col1, trait_col2 = st.columns(2)
        trait_col1.metric("Estimated Weight", f"{result['estimated_weight_kg']} kg")
        trait_col2.metric("Body Condition Score", f"{result['body_condition_score']} / 5.0")

        # ── JSON output ──────────────────────────────────────────────────
        with st.expander("Full JSON Output"):
            output = {
                "cow_detected": result["cow_detected"],
                "detection_confidence": result["detection_confidence"],
                "bbox": result["bbox"],
                **features,
                "estimated_weight_kg": result["estimated_weight_kg"],
                "body_condition_score": result["body_condition_score"],
            }
            st.json(output)
