import os
import pandas as pd
from radiomics import featureextractor

BASE_DIR = r"C:\Users\sijum\OneDrive\Documents\Pancreatic Cancer Project - Trial\processed_dataset"

IMAGE_DIR = r"C:\Users\sijum\OneDrive\Documents\Pancreatic Cancer Project - Trial\data\Task07_Pancreas\imagesTr"
MASK_DIR = r"C:\Users\sijum\OneDrive\Documents\Pancreatic Cancer Project - Trial\data\Task07_Pancreas\labelsTr"

OUTPUT_CSV = os.path.join(BASE_DIR, "radiomics_features.csv")

# ✅ INIT
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['label'] = 2  # tumor
extractor.settings['additionalInfo'] = False

print("Using label:", extractor.settings['label'])

# ✅ LOAD FILES (ignore mac junk)
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".nii.gz") and not f.startswith("._")
]

mask_files = set([
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".nii.gz") and not f.startswith("._")
])

print(f"Images: {len(image_files)}")
print(f"Masks: {len(mask_files)}")

all_features = []

# ✅ LOOP
for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, image_file)

    if image_file not in mask_files:
        print(f"❌ No mask for {image_file}")
        continue

    print(f"Processing: {image_file}")

    try:
        result = extractor.execute(image_path, mask_path)

        features = {
            k: v for k, v in result.items()
            if k.startswith("original_")
        }

        if len(features) == 0:
            print("⚠️ No features → skipping")
            continue

        features["filename"] = image_file
        all_features.append(features)

    except Exception as e:
        print(f"❌ Error: {e}")

# ✅ SAVE
print("\nTotal valid samples:", len(all_features))

if len(all_features) == 0:
    print("🚨 No features extracted. CSV not created.")
else:
    os.makedirs(BASE_DIR, exist_ok=True)

    df = pd.DataFrame(all_features)
    print("Saving CSV with shape:", df.shape)

    df.to_csv(OUTPUT_CSV, index=False)

    print("✅ CSV CREATED HERE:")
    print(OUTPUT_CSV)