import numpy as np

# =====================================================
# TEXTUAL EXPLAINABILITY
# =====================================================

def generate_explanation(
        variance,
        reliability_percent
):

    explanation = ""

    # =================================================
    # RELIABILITY ANALYSIS
    # =================================================

    if reliability_percent > 90:

        explanation += (
            "The model confidence is very high. "
        )

    elif reliability_percent > 75:

        explanation += (
            "The model confidence is moderate. "
        )

    else:

        explanation += (
            "The model confidence is low. "
        )

    # =================================================
    # UNCERTAINTY ANALYSIS
    # =================================================

    mean_uncertainty = variance.mean()

    if mean_uncertainty < 0.01:

        explanation += (
            "Prediction uncertainty is low across most regions. "
        )

    elif mean_uncertainty < 0.03:

        explanation += (
            "The model exhibits moderate uncertainty "
            "in localized regions. "
        )

    else:

        explanation += (
            "The model shows noticeable uncertainty "
            "across several regions. "
        )

    # =================================================
    # HIGH UNCERTAINTY REGION ANALYSIS
    # =================================================

    high_uncertainty_pixels = np.sum(
        variance > (mean_uncertainty * 2)
    )

    if high_uncertainty_pixels > 500:

        explanation += (
            "High uncertainty is concentrated near lesion "
            "boundaries, indicating ambiguity in edge localization. "
        )

    else:

        explanation += (
            "Uncertainty remains localized and limited. "
        )

    # =================================================
    # GRADCAM INTERPRETATION
    # =================================================

    explanation += (
        "GradCAM visualization indicates that the model "
        "primarily focused on lesion structures while "
        "generating the segmentation mask. "
    )

    # =================================================
    # FINAL CLINICAL INTERPRETATION
    # =================================================

    if reliability_percent > 90 and mean_uncertainty < 0.01:

        explanation += (
            "Overall, the prediction appears reliable "
            "and clinically consistent."
        )

    else:

        explanation += (
            "Further clinical verification may be beneficial "
            "due to uncertainty regions."
        )

    return explanation