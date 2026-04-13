import nibabel as nib
import pandas as pd
from radiomics import featureextractor
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("../processed_dataset/xgboost_cancer_model.pkl")
scaler = joblib.load("../processed_dataset/xgboost_scaler.pkl")

# Load new image + mask
IMAGE_PATH = "new_ct_image.nii.gz"
MASK_PATH = "new_mask.nii.gz"

extractor = featureextractor.RadiomicsFeatureExtractor()
result = extractor.execute(IMAGE_PATH, MASK_PATH)
features = {k: v for k, v in result.items() if k.startswith("original_")}

df = pd.DataFrame([features])
X = scaler.transform(df)

# Predict
prob = model.predict_proba(X)[:, 1][0]
print(f"Cancer risk probability: {prob:.4f}")