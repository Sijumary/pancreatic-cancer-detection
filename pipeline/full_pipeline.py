import pandas as pd
import joblib

from immunotherapy.immune_pipeline import run_immune_optimization

# 📂 Paths
FEATURES_CSV = "processed_dataset/radiomics_features.csv"
MODEL_PATH = "models/xgboost_model.pkl"

# 🔥 Load trained XGBoost model ONCE
model = joblib.load(MODEL_PATH)


def predict_cancer_risk(features_row: dict) -> float:
    """
    Convert single radiomics row into model input
    and return cancer probability
    """
    df = pd.DataFrame([features_row])

    # Ensure only numeric columns (important!)
    df = df.select_dtypes(include=["number"])

    # Fill missing values (important!)
    df = df.fillna(0)

    prob = model.predict_proba(df)[0][1]
    return float(prob)


def main():
    print("\n=== FULL AI PIPELINE START ===\n")

    df = pd.read_csv(FEATURES_CSV)

    for i, row in df.iterrows():
        print(f"\n--- Patient {i+1} ---")

        radiomics_features = row.to_dict()

        # 🔥 REAL ML prediction
        cancer_risk = predict_cancer_risk(radiomics_features)

        result = run_immune_optimization(
            radiomics_features=radiomics_features,
            cancer_risk_probability=cancer_risk
        )

        print("\nFinal Recommendation:")
        print(result)


if __name__ == "__main__":
    main()