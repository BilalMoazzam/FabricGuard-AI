
from pathlib import Path
import csv
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "aitex"

DEFECT_DIR = DATASET_DIR / "Defect_images"
MASK_DIR = DATASET_DIR / "Mask_images"
NORMAL_DIR = DATASET_DIR / "NODefect_images"

REPORT_DIR = PROJECT_ROOT / "reports" / "experiments"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. FIND DATASET FILES
# --------------------------------------------------

defect_images = sorted(DEFECT_DIR.glob("*.png"))
mask_images = sorted(MASK_DIR.glob("*.png"))
normal_images = sorted(NORMAL_DIR.rglob("*.png"))

print("\n========== AITEX DATASET INVENTORY ==========")

print(f"Defect images : {len(defect_images)}")
print(f"Mask images   : {len(mask_images)}")
print(f"Normal images : {len(normal_images)}")

print(f"Total image files: {len(defect_images) + len(mask_images) + len(normal_images)}")


# --------------------------------------------------
# 3. MATCH DEFECT IMAGES WITH MASKS
# --------------------------------------------------

def get_mask_base_name(filename):
    """
    Convert:
        0001_002_00_mask.png  -> 0001_002_00
        0044_019_04_mask1.png -> 0044_019_04
        0044_019_04_mask2.png -> 0044_019_04
    """

    stem = Path(filename).stem

    return re.sub(r"_mask\d*$", "", stem)


mask_map = {}

for mask_path in mask_images:
    base_name = get_mask_base_name(mask_path.name)

    mask_map.setdefault(base_name, []).append(mask_path)


print("\n========== MASK MATCHING ==========")

matched_count = 0
missing_masks = []
multiple_masks = []

for image_path in defect_images:
    image_name = image_path.stem

    matching_masks = mask_map.get(image_name, [])

    if matching_masks:
        matched_count += 1
    else:
        missing_masks.append(image_path.name)

    if len(matching_masks) > 1:
        multiple_masks.append(
            (image_path.name, [m.name for m in matching_masks])
        )


print(f"Defect images with masks: {matched_count}")
print(f"Defect images without masks: {len(missing_masks)}")

print("\nMissing masks:")

for filename in missing_masks:
    print(" -", filename)

print("\nImages with multiple masks:")

for image_name, masks in multiple_masks:
    print(" -", image_name)
    for mask in masks:
        print("    ", mask)


# --------------------------------------------------
# 4. INSPECT IMAGE DIMENSIONS AND MASK VALUES
# --------------------------------------------------

defect_dimensions = set()
normal_dimensions = set()
mask_dimensions = set()

for image_path in defect_images:
    image = cv2.imread(str(image_path))

    if image is not None:
        height, width = image.shape[:2]
        defect_dimensions.add((width, height))

for image_path in normal_images:
    image = cv2.imread(str(image_path))

    if image is not None:
        height, width = image.shape[:2]
        normal_dimensions.add((width, height))

for mask_path in mask_images:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is not None:
        height, width = mask.shape[:2]
        mask_dimensions.add((width, height))


print("\n========== IMAGE DIMENSIONS ==========")

print("Defect image dimensions:", defect_dimensions)
print("Normal image dimensions:", normal_dimensions)
print("Mask dimensions:", mask_dimensions)


print("\n========== MASK PIXEL VALUES ==========")

for mask_path in mask_images[:5]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is not None:
        print(
            mask_path.name,
            "Unique values:",
            np.unique(mask)[:20]
        )


# --------------------------------------------------
# 5. SAVE DATASET INVENTORY CSV
# --------------------------------------------------

csv_path = REPORT_DIR / "aitex_inventory.csv"

rows = []

for image_path in defect_images:
    image = cv2.imread(str(image_path))

    if image is None:
        print("WARNING: Cannot read", image_path.name)
        continue

    height, width = image.shape[:2]
    matching_masks = mask_map.get(image_path.stem, [])

    total_mask_pixels = 0
    dimensions_match = True

    for mask_path in matching_masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            dimensions_match = False
            continue

        if mask.shape != image.shape[:2]:
            dimensions_match = False
        else:
            total_mask_pixels += int(np.count_nonzero(mask))

    rows.append({
        "image_type": "defect",
        "filename": image_path.name,
        "fabric_group": "",
        "width": width,
        "height": height,
        "mask_count": len(matching_masks),
        "mask_files": ";".join(m.name for m in matching_masks),
        "mask_pixels": total_mask_pixels,
        "dimensions_match": dimensions_match
    })


for image_path in normal_images:
    image = cv2.imread(str(image_path))

    if image is None:
        print("WARNING: Cannot read", image_path.name)
        continue

    height, width = image.shape[:2]

    rows.append({
        "image_type": "normal",
        "filename": image_path.name,
        "fabric_group": image_path.parent.name,
        "width": width,
        "height": height,
        "mask_count": 0,
        "mask_files": "",
        "mask_pixels": 0,
        "dimensions_match": ""
    })


with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nInventory CSV saved:", csv_path)


# --------------------------------------------------
# 6. VISUALIZE SAMPLE FABRIC IMAGES
# --------------------------------------------------

def show_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_sample_figure():

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    fig.suptitle(
        "FabricGuard AI - AITEX Dataset Samples",
        fontsize=16
    )

    # Normal fabric samples
    for index, image_path in enumerate(normal_images[:3]):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        axes[0, index].imshow(show_rgb(image))
        axes[0, index].set_title(
            f"Normal: {image_path.name}",
            fontsize=9
        )
        axes[0, index].axis("off")

    # Defective fabric samples with mask overlays
    for index, image_path in enumerate(defect_images[:3]):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        display_image = show_rgb(image)
        matching_masks = mask_map.get(image_path.stem, [])

        if matching_masks:

            combined_mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8
            )

            for mask_path in matching_masks:

                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if mask is None:
                    continue

                if mask.shape == combined_mask.shape:
                    combined_mask = cv2.bitwise_or(
                        combined_mask,
                        mask
                    )

            # Draw mask boundaries for visualization
            binary_mask = (combined_mask > 0).astype(np.uint8)

            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            overlay = display_image.copy()

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (255, 0, 0),
                2
            )

            display_image = overlay

        axes[1, index].imshow(display_image)
        axes[1, index].set_title(
            f"Defect: {image_path.name}",
            fontsize=9
        )
        axes[1, index].axis("off")

    axes[0, 0].set_ylabel("Normal fabric", fontsize=12)
    axes[1, 0].set_ylabel("Defective fabric", fontsize=12)

    plt.tight_layout()

    output_path = FIGURE_DIR / "aitex_samples.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    print("\nSample visualization saved:", output_path)

    plt.show()


create_sample_figure()


# --------------------------------------------------
# 7. FINAL SUMMARY
# --------------------------------------------------

print("\n========== AUDIT COMPLETE ==========")

print("Defect :", len(defect_images))
print("Normal images:", len(normal_images))
print("Mask files:", len(mask_images))
print("Matched defect images:", matched_count)
print("Missing masks:", len(missing_masks))
print("Multiple-mask images:", len(multiple_masks))

print("\nNext step: Review the CSV and sample visualization.")
