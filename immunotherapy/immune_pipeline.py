from immunotherapy.tme_score import compute_tme_hostility_score
from immunotherapy.antigen_predictor import predict_best_antigen
from immunotherapy.crs_risk import estimate_crs_risk


def run_immune_optimization(
    radiomics_features: dict,
    cancer_risk_probability: float,
) -> dict:
    """
    Full immune optimization pipeline.

    Args:
        radiomics_features: output from your radiomics_feature_extraction.py
        cancer_risk_probability: XGBoost model output (0.0 to 1.0)

    Returns:
        Complete immunotherapy recommendation report
    """
    print("\n=== IMMUNE RESPONSE OPTIMIZATION MODULE ===\n")

    # Step 1: TME hostility
    tme = compute_tme_hostility_score(radiomics_features)
    print(f"[1] TME Hostility Score: {tme['tme_hostility_score']}")
    print(f"    → {tme['interpretation']}\n")

    # Step 2: Best antigen target
    antigens = predict_best_antigen(radiomics_features)
    top_antigen = antigens[0]
    print(f"[2] Recommended CAR-T Target: {top_antigen['antigen']}")
    print(f"    Match Score: {top_antigen['match_score']}")
    print(f"    CAR Generation: {top_antigen['car_generation']}")
    print(f"    Side Effect Risk: {top_antigen['side_effect_risk']}\n")

    # Step 3: CRS risk
    crs = estimate_crs_risk(
        cancer_risk_probability=cancer_risk_probability,
        tme_score=tme["tme_hostility_score"],
        selected_antigen=top_antigen["antigen"],
    )
    print(f"[3] Cytokine Release Syndrome Risk: {crs['crs_grade']}")
    print(f"    → {crs['management_recommendation']}\n")

    return {
        "cancer_risk_probability": cancer_risk_probability,
        "tme_analysis": tme,
        "antigen_targets": antigens,
        "top_antigen": top_antigen,
        "crs_risk": crs,
        "disclaimer": (
            "Research use only. Not for clinical decision-making. "
            "All recommendations require validation by an oncology team."
        ),
    }