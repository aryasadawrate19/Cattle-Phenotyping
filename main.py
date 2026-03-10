"""
Livestock Phenotyping MVP — CLI Entry Point
Run the phenotyping pipeline on a single image from the command line.

Usage:
    python main.py --image path/to/cow.jpg
"""

import argparse
import json
import sys

import cv2

from pipeline.phenotyping_pipeline import PhenotypingPipeline
from utils.visualization import draw_bbox, overlay_mask


def main():
    parser = argparse.ArgumentParser(
        description="Livestock Phenotyping MVP — estimate cattle traits from an image"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to cow image")
    parser.add_argument(
        "--save_vis",
        type=str,
        default=None,
        help="Path to save annotated output image (optional)",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: could not read image at {args.image}")
        sys.exit(1)

    pipeline = PhenotypingPipeline()
    result = pipeline.run(image)

    if not result["cow_detected"]:
        print("No cow detected in the image.")
        sys.exit(0)

    # Build output dict
    output = {
        "cow_detected": result["cow_detected"],
        "detection_confidence": result["detection_confidence"],
        "bbox": result["bbox"],
        **result["features"],
        "estimated_weight_kg": result["estimated_weight_kg"],
        "body_condition_score": result["body_condition_score"],
    }

    print(json.dumps(output, indent=2))

    # Optionally save annotated image
    if args.save_vis:
        vis = draw_bbox(image, result["bbox"])
        vis = overlay_mask(vis, result["mask"])
        cv2.imwrite(args.save_vis, vis)
        print(f"Annotated image saved to {args.save_vis}")


if __name__ == "__main__":
    main()
