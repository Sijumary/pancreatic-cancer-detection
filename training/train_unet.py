import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import PancreasDataset

# -------------------------------
# Paths to dataset
# -------------------------------
IMAGE_DIR = "processed_dataset/images"
MASK_DIR = "processed_dataset/masks"

# -------------------------------
# Dataset and DataLoader
# -------------------------------
dataset = PancreasDataset(IMAGE_DIR, MASK_DIR)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

print(f"Total training samples: {len(dataset)}", flush=True)

# -------------------------------
# Model: U-Net with ResNet34 encoder
# -------------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,  # grayscale images
    classes=1
)

# -------------------------------
# Device: CPU or GPU
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}", flush=True)

# -------------------------------
# Optimizer & Loss
# -------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.BCEWithLogitsLoss()

# -------------------------------
# Training parameters
# -------------------------------
EPOCHS = 5

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)

        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}",
                flush=True
            )

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}", flush=True)

# -------------------------------
# Save trained model
# -------------------------------
os.makedirs("training/models", exist_ok=True)
model_path = "training/models/pancreas_unet_windows.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}", flush=True)