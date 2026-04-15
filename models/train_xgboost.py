import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load radiomics features
DATA_PATH = "../processed_dataset/radiomics_features.csv"

df = pd.read_csv(DATA_PATH)

# You NEED a target column
# Example: 'label' (0 = normal, 1 = cancer)
if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column")

# Split features & target
X = df.drop(columns=["label"])
y = df["label"]

df["label"] = np.random.randint(0, 2, size=len(df))

# Keep only numeric data
X = X.select_dtypes(include=["number"]).fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc:.2f}")

# Save model + feature columns
joblib.dump((model, X.columns.tolist()), "xgboost_model.pkl")

print("Model saved to models/xgboost_model.pkl")