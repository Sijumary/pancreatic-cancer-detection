import os
import nibabel as nib
import numpy as np
import cv2

IMAGE_DIR = "data/Task07_Pancreas/imagesTr"
LABEL_DIR = "data/Task07_Pancreas/labelsTr"

OUT_IMG = "tumor_dataset/images"
OUT_MASK = "tumor_dataset/masks"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

slice_id = 0

for file in os.listdir(IMAGE_DIR):

    # Skip macOS / hidden files
    if file.startswith("._") or file.startswith("."):
        continue

    # Process only NIfTI images
    if not file.endswith(".nii.gz"):
        continue

    img_path = os.path.join(IMAGE_DIR, file)
    label_path = os.path.join(LABEL_DIR, file)

    print("Processing:", file)

    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    for i in range(img.shape[2]):

        image_slice = img[:, :, i]
        label_slice = label[:, :, i]

        # tumor mask
        tumor_mask = (label_slice == 2).astype(np.uint8)

        if tumor_mask.sum() == 0:
            continue

        image_slice = cv2.normalize(image_slice, None, 0, 255, cv2.NORM_MINMAX)

        cv2.imwrite(f"{OUT_IMG}/{slice_id}.png", image_slice)
        cv2.imwrite(f"{OUT_MASK}/{slice_id}.png", tumor_mask * 255)

        slice_id += 1

print("Tumor dataset slices:", slice_id)