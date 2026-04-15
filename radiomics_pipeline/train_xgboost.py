import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

# Paths
FEATURE_CSV = "../processed_dataset/radiomics_features.csv"

# Load features
df = pd.read_csv(FEATURE_CSV)

# Assume you have a column "label" where 1 = cancer risk, 0 = normal
# If not, you must add labels manually
if "label" not in df.columns:
    raise ValueError("Add 'label' column to CSV for classification")

X = df.drop(columns=["filename", "label"])
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, "../processed_dataset/xgboost_cancer_model.pkl")
print("Model saved to xgboost_cancer_model.pkl")