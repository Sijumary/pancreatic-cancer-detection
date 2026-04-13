import nibabel as nib
import matplotlib.pyplot as plt

path = "../data/Task07_Pancreas/imagesTr/pancreas_001.nii.gz"

img = nib.load(path)
data = img.get_fdata()

print("Shape:", data.shape)

slice_index = data.shape[2] // 2

plt.imshow(data[:, :, slice_index], cmap="gray")
plt.title("CT Scan Slice")
plt.show()