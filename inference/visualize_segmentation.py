import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# ---- PATHS ----
BASE_DIR = r"C:\Users\sijum\OneDrive\Documents\Pancreatic Cancer Project - Trial"

IMAGE_PATH = BASE_DIR + r"\data\Task07_Pancreas\imagesTr\pancreas_001.nii.gz"
MASK_PATH  = BASE_DIR + r"\data\Task07_Pancreas\labelsTr\pancreas_001.nii.gz"


img = nib.load(IMAGE_PATH).get_fdata()
mask = nib.load(MASK_PATH).get_fdata()

# pick middle slice
slice_idx = img.shape[2] // 2

image_slice = img[:, :, slice_idx]
mask_slice = (mask[:, :, slice_idx] == 2)

# ---- PLOT ----
plt.figure(figsize=(8, 6))
plt.imshow(image_slice, cmap="gray")
plt.imshow(mask_slice, cmap="jet", alpha=0.4)
plt.title("Pancreas Segmentation")
plt.axis("off")

# ---- SAVE ----
os.makedirs("screenshots", exist_ok=True)
plt.savefig("screenshots/segmentation.png")
plt.close()

print("Saved: screenshots/segmentation.png")