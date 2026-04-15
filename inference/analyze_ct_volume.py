import nibabel as nib
import numpy as np
import torch
import cv2
import os
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# -----------------------
# CREATE OUTPUT FOLDER
# -----------------------

os.makedirs("outputs", exist_ok=True)

# -----------------------
# MODEL PATHS
# -----------------------

PANCREAS_MODEL = "training/models/pancreas_unet_windows.pth"
TUMOR_MODEL = "training/models/tumor_unet.pth"

# -----------------------
# DEVICE
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# LOAD MODELS
# -----------------------

pancreas_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

tumor_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

pancreas_model.load_state_dict(torch.load(PANCREAS_MODEL, map_location=device))
tumor_model.load_state_dict(torch.load(TUMOR_MODEL, map_location=device))

pancreas_model.to(device)
tumor_model.to(device)

pancreas_model.eval()
tumor_model.eval()

print("Models loaded")

# -----------------------
# LOAD CT VOLUME
# -----------------------

CT_PATH = "data/Task07_Pancreas/imagesTr/pancreas_001.nii.gz"

ct = nib.load(CT_PATH).get_fdata()

print("CT Volume Shape:", ct.shape)

tumor_volume = np.zeros(ct.shape)

# -----------------------
# PROCESS EACH SLICE
# -----------------------

for i in range(ct.shape[2]):

    slice_img = ct[:, :, i]

    # normalize slice
    slice_img = cv2.normalize(slice_img, None, 0, 1, cv2.NORM_MINMAX)

    tensor = torch.tensor(slice_img).unsqueeze(0).unsqueeze(0).float().to(device)

    # pancreas prediction
    with torch.no_grad():
        pancreas_pred = torch.sigmoid(pancreas_model(tensor))

    pancreas_mask = pancreas_pred.cpu().numpy()[0][0]

    # tumor prediction
    with torch.no_grad():
        tumor_pred = torch.sigmoid(tumor_model(tensor))

    tumor_mask = tumor_pred.cpu().numpy()[0][0]

    # restrict tumor detection inside pancreas
    tumor_mask = tumor_mask * pancreas_mask

    tumor_volume[:, :, i] = tumor_mask

print("3D tumor volume reconstructed")

# -----------------------
# STEP 4: TUMOR VOLUME ANALYSIS
# -----------------------

tumor_voxels = np.sum(tumor_volume > 0.5)

print("Tumor voxel count:", tumor_voxels)

# estimate tumor volume (voxel size assumed ~1mm³ for demo)
tumor_volume_mm3 = tumor_voxels
tumor_volume_ml = tumor_volume_mm3 / 1000

print("Estimated tumor volume (mm³):", tumor_volume_mm3)
print("Estimated tumor volume (mL):", tumor_volume_ml)

# -----------------------
# STEP 5: SAVE 3D TUMOR MASK
# -----------------------

output_path = "outputs/tumor_prediction.nii.gz"

nib.save(
    nib.Nifti1Image(tumor_volume, np.eye(4)),
    output_path
)

print("3D tumor mask saved to:", output_path)

# -----------------------
# STEP 6: VISUALIZATION
# -----------------------

slice_index = ct.shape[2] // 2

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("CT Slice")
plt.imshow(ct[:,:,slice_index], cmap="gray")

plt.subplot(1,3,2)
plt.title("Tumor Prediction")
plt.imshow(tumor_volume[:,:,slice_index], cmap="hot")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(ct[:,:,slice_index], cmap="gray")
plt.imshow(tumor_volume[:,:,slice_index], alpha=0.5, cmap="jet")

plt.show()

# -----------------------
# OPTIONAL: 3D VISUALIZATION
# -----------------------

try:
    from vedo import Volume

    print("Launching 3D tumor visualization...")

    vol = Volume(tumor_volume)
    vol.show()

except Exception:
    print("Vedo not installed. Run: pip install vedo for 3D visualization.")
