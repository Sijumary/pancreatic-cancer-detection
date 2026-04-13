import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from tumor_dataset import TumorDataset

IMAGE_DIR = "tumor_dataset/images"
MASK_DIR = "tumor_dataset/masks"

dataset = TumorDataset(IMAGE_DIR, MASK_DIR)

train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss_fn = torch.nn.BCEWithLogitsLoss()

EPOCHS = 10

for epoch in range(EPOCHS):

    total_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch+1, "Loss:", total_loss/len(train_loader), flush=True)

torch.save(model.state_dict(), "training/models/tumor_unet.pth")

print("Tumor model trained!")