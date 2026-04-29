"""
Audit labels and image coverage before training or writing paper results.

Usage:
    python training/audit_dataset.py --data_dir data/images --labels data/labels.csv
"""

import argparse
import collections
import csv
import os


MEASUREMENT_COLUMNS = [
    "weight",
    "body_length_cm",
    "withers_height_cm",
    "heart_girth_cm",
    "hip_length_cm",
]


def main():
    parser = argparse.ArgumentParser(description="Audit cattle phenotyping labels")
    parser.add_argument("--data_dir", default="data/images")
    parser.add_argument("--labels", default="data/labels.csv")
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))

    image_names = {
        name
        for name in os.listdir(args.data_dir)
        if os.path.isfile(os.path.join(args.data_dir, name))
    }
    labelled_names = {row["image_name"] for row in rows}

    print(f"Label rows: {len(rows)}")
    print(f"Image files: {len(image_names)}")
    print(f"Missing labelled images: {sorted(labelled_names - image_names)}")
    print(f"Unlabelled images: {sorted(image_names - labelled_names)}")

    bcs_counts = collections.Counter(row["bcs"] for row in rows)
    print("BCS distribution:")
    for bcs, count in sorted(bcs_counts.items(), key=lambda item: float(item[0])):
        print(f"  {bcs}: {count}")

    grouped = collections.defaultdict(list)
    for row in rows:
        key = tuple(row[column] for column in MEASUREMENT_COLUMNS)
        grouped[key].append(row["image_name"])

    duplicate_groups = [names for names in grouped.values() if len(names) > 1]
    print(f"Duplicate measurement groups: {len(duplicate_groups)}")
    print(f"Rows in duplicate groups: {sum(len(names) for names in duplicate_groups)}")

    if "animal_id" not in rows[0]:
        print("Recommendation: add animal_id if duplicate groups are repeated images of the same animal.")


if __name__ == "__main__":
    main()
