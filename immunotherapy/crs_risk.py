def estimate_crs_risk(
    cancer_risk_probability: float,  # output from your XGBoost model (0-1)
    tme_score: float,                # from tme_score.py
    selected_antigen: str,
) -> dict:
    """
    Args:
        cancer_risk_probability: XGBoost output (higher = more tumour burden)
        tme_score: TME hostility score (higher = more stroma)
        selected_antigen: top antigen from antigen_predictor
    Returns:
        CRS risk level, management recommendation
    """
    # Higher tumour burden + lower stroma barrier = more T cells activated = higher CRS
    antigen_crs_weights = {
        "Mesothelin (MSLN)": 0.7,
        "Claudin 18.2 (CLDN18.2)": 0.3,
        "MUC1": 0.4,
        "CEA (Carcinoembryonic Antigen)": 0.3,
    }

    antigen_weight = antigen_crs_weights.get(selected_antigen, 0.5)
    infiltration_ease = 1 - tme_score  # easy infiltration = more T cell activation

    crs_score = (
        0.5 * cancer_risk_probability +
        0.3 * infiltration_ease +
        0.2 * antigen_weight
    )

    if crs_score > 0.65:
        grade = "HIGH"
        management = (
            "Start tocilizumab (IL-6 inhibitor) prophylactically. "
            "Consider dose-fractionated CAR-T infusion. "
            "ICU monitoring recommended for first 72 hours post-infusion."
        )
    elif crs_score > 0.35:
        grade = "MODERATE"
        management = (
            "Monitor cytokine levels (IL-6, ferritin, CRP) for 48 hours. "
            "Corticosteroid standby. "
            "Outpatient infusion possible with daily follow-up."
        )
    else:
        grade = "LOW"
        management = (
            "Standard monitoring protocol. "
            "Low risk of severe immune reaction — favourable for outpatient setting."
        )

    return {
        "crs_risk_score": round(crs_score, 3),
        "crs_grade": grade,
        "management_recommendation": management,
    }