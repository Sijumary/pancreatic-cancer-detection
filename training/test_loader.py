from torch.utils.data import DataLoader
from dataset import PancreasDataset

dataset = PancreasDataset("processed_dataset/images", "processed_dataset/masks")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

for images, masks in loader:
    print("Images batch shape:", images.shape)
    print("Masks batch shape:", masks.shape)
    break