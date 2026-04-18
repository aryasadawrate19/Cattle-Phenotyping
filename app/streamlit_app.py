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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.phenotyping_pipeline import PhenotypingPipeline
from utils.visualization import draw_bbox, overlay_mask, create_mask_visualization

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cattle Phenotyping System",
    page_icon="🐄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .big-metric {
        background: #f0f7f0;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .big-metric .label {
        font-size: 14px;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .big-metric .value {
        font-size: 42px;
        font-weight: 700;
        color: #1b5e20;
        line-height: 1.1;
    }
    .big-metric .sub {
        font-size: 13px;
        color: #777;
        margin-top: 4px;
    }
    .bcs-bar-container {
        background: #e8f5e9;
        border-left: 5px solid #388e3c;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .pipeline-step {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
    }
    .model-perf {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #444;
    }
    .warn-box {
        background: #fff3e0;
        border-left: 4px solid #ef6c00;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #555;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models — YOLOv8, SAM, XGBoost...")
def load_pipeline():
    return PhenotypingPipeline()


@st.cache_data
def load_cv_results():
    path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "cv_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def bcs_color(bcs: float) -> str:
    if bcs < 2.0:
        return "#e53935"   # red — very thin
    elif bcs < 2.5:
        return "#fb8c00"   # orange — thin
    elif bcs <= 3.5:
        return "#43a047"   # green — ideal
    elif bcs <= 4.0:
        return "#fb8c00"   # orange — overweight
    else:
        return "#e53935"   # red — obese


def bcs_label(bcs: float) -> str:
    if bcs < 2.0:
        return "Very thin"
    elif bcs < 2.5:
        return "Thin"
    elif bcs <= 3.5:
        return "Ideal"
    elif bcs <= 4.0:
        return "Overweight"
    else:
        return "Obese"


pipeline = load_pipeline()
cv_results = load_cv_results()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🐄 Cattle Phenotyping System")
st.markdown("Estimate **body weight**, **body condition score**, and **morphometric measurements** from a single side-view image.")

# ── Model performance banner ──────────────────────────────────────────────────
if cv_results and "summary" in cv_results:
    s = cv_results["summary"]
    st.markdown(f"""
    <div class="model-perf">
        <b>Model performance (5-fold cross-validation, n=72):</b> &nbsp;
        Weight MAE = <b>{s['weight_mae_mean']:.1f} ± {s['weight_mae_std']:.1f} kg</b>
        &nbsp;|&nbsp; Weight R² = <b>{s['weight_r2_mean']:.3f}</b>
        &nbsp;|&nbsp; BCS MAE = <b>{s['bcs_mae_mean']:.2f} ± {s['bcs_mae_std']:.2f}</b>
        &nbsp;|&nbsp; BCS R² = <b>{s['bcs_r2_mean']:.3f}</b>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Display Options")
    show_mask_overlay = st.checkbox("Show segmentation overlay", value=True)
    show_raw_mask = st.checkbox("Show raw mask", value=False)
    show_pixel_features = st.checkbox("Show pixel-level features", value=False)

    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    **Pipeline:**
    1. YOLOv8n — cow detection
    2. SAM ViT-B — body segmentation
    3. OpenCV — morphological features
    4. XGBoost — trait prediction

    **Dataset:** 72 Horqin yellow cattle  
    **Training:** 5-fold cross-validation
    """)

    trained = pipeline.predictor.weight_model is not None
    if trained:
        st.success("✅ Trained XGBoost models loaded")
    else:
        st.warning("⚠️ No trained models found — using heuristics")

    st.divider()
    if st.button("🔄 Reload Models"):
        st.cache_resource.clear()
        st.rerun()


# ── Image upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a side-view cow image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="For best results: clear side-view, single animal, natural lighting",
)

if uploaded_file is None:
    st.info("👆 Upload an image to begin. The pipeline will detect, segment, and measure the animal automatically.")
    st.stop()

# ── Read & display uploaded image ─────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Running pipeline — detection → segmentation → feature extraction → prediction..."):
    result = pipeline.run(image_bgr)

if not result["cow_detected"]:
    st.error("❌ No cow detected in this image. Please upload a clear side-view cattle photo.")
    st.image(image_rgb, caption="Uploaded image", use_container_width=True)
    st.stop()

# ── Layout: results ───────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📷 Pipeline Output")

    tab1, tab2 = st.tabs(["Detection", "Segmentation"])

    with tab1:
        bbox_vis = draw_bbox(result["processed_image"], result["bbox"])
        st.image(cv2.cvtColor(bbox_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Detection confidence: {result['detection_confidence']:.1%}")

    with tab2:
        if show_mask_overlay:
            mask_vis = overlay_mask(result["processed_image"], result["mask"])
            st.image(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        if show_raw_mask:
            st.image(create_mask_visualization(result["mask"]), use_container_width=True)
        if not show_mask_overlay and not show_raw_mask:
            st.info("Enable an overlay option in the sidebar to see the mask.")

with right:
    st.subheader("📊 Predicted Traits")

    weight = result["estimated_weight_kg"]
    bcs = result["body_condition_score"]
    color = bcs_color(bcs)
    label = bcs_label(bcs)

    # Weight
    st.markdown(f"""
    <div class="big-metric">
        <div class="label">Estimated Body Weight</div>
        <div class="value">{weight} kg</div>
        <div class="sub">Model MAE ≈ 16 kg &nbsp;|&nbsp; ~{weight * 2.205:.0f} lbs</div>
    </div>
    """, unsafe_allow_html=True)

    # BCS with visual bar
    filled = int(round(bcs))
    dots = ""
    for i in range(1, 6):
        dots += f"<span style='font-size:22px; color:{'#333' if i <= filled else '#ccc'}'>●</span> "

    st.markdown(f"""
    <div class="bcs-bar-container">
        <div class="label" style="color:#555; font-size:14px; text-transform:uppercase; letter-spacing:0.05em;">Body Condition Score</div>
        <div style="font-size:42px; font-weight:700; color:{color}; line-height:1.1;">{bcs} <span style="font-size:18px; color:#888;">/ 5.0</span></div>
        <div style="margin: 6px 0;">{dots}</div>
        <div style="font-size:13px; color:{color}; font-weight:600;">{label}</div>
        <div style="font-size:12px; color:#888; margin-top:4px;">Model MAE ≈ 0.28 score units &nbsp;|&nbsp; Ideal range: 2.5 – 3.5</div>
    </div>
    """, unsafe_allow_html=True)

    # Morphometric measurements
    st.subheader("📐 Body Measurements")
    m1, m2 = st.columns(2)
    m1.metric("Body Length", f"{result.get('body_length_cm', 0):.1f} cm")
    m2.metric("Withers Height", f"{result.get('withers_height_cm', 0):.1f} cm")
    m3, m4 = st.columns(2)
    m3.metric("Heart Girth", f"{result.get('heart_girth_cm', 0):.1f} cm")
    m4.metric("Hip Length", f"{result.get('hip_length_cm', 0):.1f} cm")

# ── Out-of-distribution warning ───────────────────────────────────────────────
if uploaded_file.name not in [f"{i}.png" for i in range(1, 73)]:
    st.markdown("""
    <div class="warn-box">
        ⚠️ <b>Note:</b> This model was trained on Horqin yellow cattle photographed at 1m distance
        under controlled conditions. Predictions on other breeds or imaging setups may be less accurate.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Pixel features (collapsed by default) ────────────────────────────────────
if show_pixel_features:
    st.subheader("🔬 Raw Pixel Features")
    features = result["features"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Body Area (px)", f"{features['body_area_px']:,}")
    c2.metric("Body Length (px)", features["body_length_px"])
    c3.metric("Body Height (px)", features["body_height_px"])
    c4.metric("Aspect Ratio", features["aspect_ratio"])
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Solidity", features["solidity"])
    c6.metric("Compactness", features["compactness"])
    c7.metric("Convex Hull (px)", f"{features['convex_hull_area']:,}")
    c8.metric("Body Area Ratio", features["body_area_ratio"])

# ── JSON export ───────────────────────────────────────────────────────────────
with st.expander("📄 Full JSON Output"):
    output = {
        "cow_detected": result["cow_detected"],
        "detection_confidence": result["detection_confidence"],
        "bbox": result["bbox"],
        "estimated_weight_kg": result["estimated_weight_kg"],
        "body_condition_score": result["body_condition_score"],
        "body_length_cm": result.get("body_length_cm", 0.0),
        "withers_height_cm": result.get("withers_height_cm", 0.0),
        "heart_girth_cm": result.get("heart_girth_cm", 0.0),
        "hip_length_cm": result.get("hip_length_cm", 0.0),
        **result["features"],
    }
    st.json(output)
    st.download_button(
        "⬇️ Download JSON",
        data=json.dumps(output, indent=2),
        file_name="phenotyping_result.json",
        mime="application/json",
    )