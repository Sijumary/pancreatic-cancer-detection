import os
import nibabel as nib
import numpy as np
import cv2

# Dataset paths
IMAGE_DIR = "../data/Task07_Pancreas/imagesTr"
MASK_DIR = "../data/Task07_Pancreas/labelsTr"

# Output dataset
OUTPUT_IMAGE_DIR = "../processed_dataset/images"
OUTPUT_MASK_DIR = "../processed_dataset/masks"

# Create output folders
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

slice_count = 0

for file in os.listdir(IMAGE_DIR):

    # Skip hidden/macOS metadata files
    if file.startswith(".") or file.startswith("._"):
        continue

    # Only process real CT scans
    if not file.endswith(".nii.gz"):
        continue

    image_path = os.path.join(IMAGE_DIR, file)
    mask_path = os.path.join(MASK_DIR, file)

    print("Processing:", file)

    try:
        img = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
    except Exception as e:
        print("Skipping corrupted file:", file)
        continue

    for i in range(img.shape[2]):

        image_slice = img[:, :, i]
        mask_slice = mask[:, :, i]

        # Skip slices with no pancreas
        if np.sum(mask_slice) == 0:
            continue

        # Normalize CT slice
        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        image_slice = (image_slice * 255).astype(np.uint8)

        image_name = f"{slice_count}.png"
        mask_name = f"{slice_count}.png"

        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, image_name), image_slice)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, mask_name), mask_slice)

        slice_count += 1

print("Total slices saved:", slice_count)
