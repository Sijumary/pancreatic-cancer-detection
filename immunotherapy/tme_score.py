import numpy as np
from immunotherapy.tme_score import compute_tme_hostility_score
from immunotherapy.antigen_predictor import predict_best_antigen
from immunotherapy.crs_risk import estimate_crs_risk

# These are radiomics feature names your PyRadiomics extraction produces.
# Adjust keys to match your actual feature names.
STROMA_PROXY_FEATURES = [
    "original_glcm_Contrast",       # high contrast = heterogeneous stroma
    "original_glcm_Correlation",    # high correlation = dense regular stroma
    "original_shape_Sphericity",    # low sphericity = irregular mass
    "original_firstorder_Entropy",  # high entropy = disorganized tissue
]

def compute_tme_hostility_score(radiomics_features: dict) -> dict:
    """
    Args:
        radiomics_features: dict of feature_name -> value from PyRadiomics
    Returns:
        dict with score (0-1), interpretation, and infiltration probability
    """
    values = []
    for feat in STROMA_PROXY_FEATURES:
        if feat in radiomics_features:
            values.append(radiomics_features[feat])

    if not values:
        return {"error": "No matching radiomics features found"}

    # Normalize to 0-1 using sigmoid
    raw = np.mean(values)
    score = 1 / (1 + np.exp(-raw))  # sigmoid normalisation

    infiltration_probability = round(1 - score, 3)

    if score > 0.7:
        interpretation = "High stromal density — CAR-T infiltration likely poor without stroma-disrupting co-therapy"
        recommendation = "Consider combining CAR-T with hyaluronidase or FAP-targeting agents to break down stroma"
    elif score > 0.4:
        interpretation = "Moderate stromal barrier — CAR-T may partially infiltrate"
        recommendation = "Standard 2nd or 4th generation CAR-T with checkpoint inhibitor combination"
    else:
        interpretation = "Low stromal density — favourable for immune cell infiltration"
        recommendation = "Direct CAR-T therapy likely effective"

    return {
        "tme_hostility_score": round(float(score), 3),
        "t_cell_infiltration_probability": infiltration_probability,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }