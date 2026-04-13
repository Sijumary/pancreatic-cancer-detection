import torch
import numpy as np
import cv2
import os
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# -----------------------
# Model configuration
# -----------------------

MODEL_PATH = "training/models/pancreas_unet_windows.pth"
IMAGE_DIR = "processed_dataset/images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load model
# -----------------------

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully")

# -----------------------
# Pick a sample image
# -----------------------

image_files = os.listdir(IMAGE_DIR)
image_path = os.path.join(IMAGE_DIR, image_files[0])

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = image / 255.0

input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)

# -----------------------
# Prediction
# -----------------------

with torch.no_grad():
    pred = model(input_tensor)
    pred = torch.sigmoid(pred)

mask = pred.cpu().numpy()[0][0]

# -----------------------
# Visualization
# -----------------------

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("CT Slice")
plt.imshow(image, cmap="gray")

plt.subplot(1,3,2)
plt.title("Predicted Mask")
plt.imshow(mask, cmap="hot")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(image, cmap="gray")
plt.imshow(mask, alpha=0.5, cmap="jet")

plt.show()