# Livestock Phenotyping MVP

A research prototype that estimates cattle traits (body condition score and weight) from a single image using computer vision and machine learning.

## Pipeline Architecture

```
Image → Cow Detection (YOLOv8n) → Segmentation (SAM ViT-B) → Feature Extraction (OpenCV) → Trait Prediction (XGBoost) → Output
```

### Models Used

| Stage | Model | Purpose |
|---|---|---|
| Detection | YOLOv8n (Ultralytics, COCO pretrained) | Detect cow bounding box |
| Segmentation | Segment Anything Model (SAM ViT-B) | Generate binary cow mask |
| Trait Prediction | XGBoost Regressor | Predict weight (kg) and BCS |

## Project Structure

```
livestock-phenotyping-mvp/
├── models/
│   ├── detector_yolov8.py       # YOLOv8 cow detector
│   ├── segmenter_sam.py         # SAM segmenter
│   └── trait_model_xgboost.py   # XGBoost trait predictor
├── pipeline/
│   ├── feature_extractor.py     # Morphological feature extraction
│   └── phenotyping_pipeline.py  # Full pipeline orchestration
├── training/
│   └── train_trait_model.py     # Train XGBoost on labelled data
├── app/
│   └── streamlit_app.py         # Streamlit web interface
├── data/
│   ├── images/                  # Place cow images here
│   └── labels.csv               # Training labels
├── utils/
│   └── visualization.py         # Drawing / overlay utilities
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SAM Checkpoint

The SAM ViT-B checkpoint (~375 MB) will be **downloaded automatically** on first run. To download manually:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Place the file in the project root directory.

### 3. YOLOv8 Weights

YOLOv8n weights are downloaded automatically by Ultralytics on first use.

## Usage

### CLI — Single Image

```bash
cd livestock-phenotyping-mvp
python main.py --image path/to/cow.jpg
```

Optional: save annotated output image:

```bash
python main.py --image path/to/cow.jpg --save_vis output.jpg
```

### Streamlit App

```bash
cd livestock-phenotyping-mvp
streamlit run app/streamlit_app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501).

### Train XGBoost Models

1. Place labelled cow images in `data/images/`.
2. Fill in `data/labels.csv` with columns: `image_name`, `weight`, `bcs`.
3. Run training:

```bash
python training/train_trait_model.py --data_dir data/images --labels data/labels.csv
```

Trained models are saved to `saved_models/`.

## Example Output

```json
{
  "cow_detected": true,
  "detection_confidence": 0.87,
  "bbox": [120, 80, 740, 450],
  "body_area_px": 152343,
  "body_length_px": 620,
  "body_height_px": 270,
  "bbox_width": 620,
  "bbox_height": 370,
  "contour_perimeter": 1842.5,
  "convex_hull_area": 165200,
  "aspect_ratio": 2.3,
  "estimated_weight_kg": 430.0,
  "body_condition_score": 3.1
}
```

## Notes

- The system works on **single side-view images** of cattle.
- Without trained XGBoost models, **heuristic estimates** are used for weight and BCS.
- For best accuracy, train the XGBoost models on your own labelled dataset.
- GPU is recommended for SAM inference but CPU works (slower).
