import os
import nibabel as nib
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\sijum\OneDrive\Documents\Pancreatic Cancer Project - Trial"

IMAGE_PATH = os.path.join(BASE_DIR, "data", "Task07_Pancreas", "imagesTr", "pancreas_001.nii.gz")
TUMOR_MASK = os.path.join(BASE_DIR, "data", "Task07_Pancreas", "labelsTr", "pancreas_001.nii.gz")

print("Image exists:", os.path.exists(IMAGE_PATH))
print("Mask exists:", os.path.exists(TUMOR_MASK))

img = nib.load(IMAGE_PATH).get_fdata()
mask = nib.load(TUMOR_MASK).get_fdata()

slice_idx = img.shape[2] // 2

plt.figure(figsize=(8, 6))
plt.imshow(img[:, :, slice_idx], cmap="gray")

# tumor = label 2 (IMPORTANT for Medical Decathlon dataset)
plt.imshow(mask[:, :, slice_idx] == 2, cmap="hot", alpha=0.5)

plt.title("Tumor Detection")
plt.axis("off")

os.makedirs("screenshots", exist_ok=True)
plt.savefig("screenshots/tumor_detection.png")

print("Saved: screenshots/tumor_detection.png")