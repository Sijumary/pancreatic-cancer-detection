import numpy as np

# Antigen profiles: each has a weight vector aligned to radiomics signals.
# These weights are literature-informed heuristics; replace with trained model
# once you have gene expression data (TCGA-PAAD) paired with imaging.

ANTIGEN_PROFILES = {
    "Mesothelin (MSLN)": {
        "description": "Overexpressed in ~80% of PDAC. Strong CAR-T target.",
        "car_generation": "4th gen (TRUCK)",
        "side_effect_risk": "Moderate — pulmonary toxicity risk at high dose",
        "intensity_affinity": 0.8,   # prefers high-intensity tumours
        "entropy_affinity": 0.6,
    },
    "Claudin 18.2 (CLDN18.2)": {
        "description": "Breakthrough 2024 target. Highly specific to pancreatic/gastric tumours.",
        "car_generation": "2nd or 3rd gen",
        "side_effect_risk": "Low — high tumour specificity reduces off-target effects",
        "intensity_affinity": 0.6,
        "entropy_affinity": 0.7,
    },
    "MUC1": {
        "description": "Widely expressed but variably glycosylated. Good for high-entropy tumours.",
        "car_generation": "3rd gen",
        "side_effect_risk": "Low-Moderate",
        "intensity_affinity": 0.5,
        "entropy_affinity": 0.9,
    },
    "CEA (Carcinoembryonic Antigen)": {
        "description": "Broadly expressed. Useful for metastatic PDAC.",
        "car_generation": "2nd gen",
        "side_effect_risk": "Low",
        "intensity_affinity": 0.4,
        "entropy_affinity": 0.5,
    },
}

def predict_best_antigen(radiomics_features: dict) -> list:
    """
    Args:
        radiomics_features: dict of radiomics feature name -> value
    Returns:
        Ranked list of antigen targets with scores
    """
    intensity = radiomics_features.get("original_firstorder_Mean", 0.5)
    entropy = radiomics_features.get("original_firstorder_Entropy", 0.5)

    # Normalise to 0-1
    norm_intensity = min(max(intensity / 500, 0), 1)  # HU scale approx
    norm_entropy = min(max(entropy / 5, 0), 1)

    scored = []
    for antigen, profile in ANTIGEN_PROFILES.items():
        score = (
            profile["intensity_affinity"] * norm_intensity +
            profile["entropy_affinity"] * norm_entropy
        ) / 2
        scored.append({
            "antigen": antigen,
            "match_score": round(score, 3),
            "car_generation": profile["car_generation"],
            "description": profile["description"],
            "side_effect_risk": profile["side_effect_risk"],
        })

    return sorted(scored, key=lambda x: x["match_score"], reverse=True)