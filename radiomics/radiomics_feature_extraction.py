import os
import pandas as pd
import nibabel as nib
from radiomics import featureextractor



# Ensure output directory exists
OUTPUT_DIR = r"C:\Users\sijum\OneDrive\Documents\Pancreatic Cancer Project - Trial\processed_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Then set your CSV path


# Paths
IMAGE_DIR = r"C:\Users\sijum\OneDrive\Documents\Pancreatic Cancer Project - Trial\processed_dataset\images"  # segmented pancreas or tumor CT images
MASK_DIR = r"C:\Users\sijum\OneDrive\Documents\Pancreatic Cancer Project - Trial\processed_dataset\masks"    # corresponding masks
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "radiomics_features.csv")

# Initialize radiomics extractor with default settings
extractor = featureextractor.RadiomicsFeatureExtractor()

# Optional: print settings
print("Settings:", extractor.settings)

# Collect data
all_features = []

for image_file in os.listdir(IMAGE_DIR):
    if not image_file.endswith(".nii.gz"):
        continue
    image_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, image_file.replace("image", "mask"))
    
    if not os.path.exists(mask_path):
        print(f"Mask not found for {image_file}, skipping...")
        continue
    
    print(f"Processing {image_file}...")
    img = nib.load(image_path)  # load CT image
    mask = nib.load(mask_path)  # load mask
    
    # Extract features
    result = extractor.execute(image_path, mask_path)
    
    # Flatten dictionary, remove non-feature entries
    features = {k: v for k, v in result.items() if k.startswith("original_")}
    features["filename"] = image_file
    
    all_features.append(features)

# Save to CSV
df = pd.DataFrame(all_features)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Radiomics features saved to {OUTPUT_CSV}")