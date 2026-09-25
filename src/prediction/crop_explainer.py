import numpy as np
import pandas as pd
import shap

from src.models.model_loader import load_model


# ============================================================
# LOAD CROP MODEL
# ============================================================

model = load_model()


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_NAMES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


# ============================================================
# CREATE SHAP EXPLANATION
# ============================================================

def explain_crop_prediction(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):
    """
    Generate SHAP feature importance values for
    a crop recommendation.
    """

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        input_data
    )

    # --------------------------------------------------------
    # CONVERT SHAP OUTPUT
    # --------------------------------------------------------

    shap_array = np.asarray(shap_values)

    # SHAP can return:
    #
    # (samples, features, classes)
    #
    # or
    #
    # (samples, features)
    #
    # depending on the SHAP version/model.

    if shap_array.ndim == 3:

        # Select the predicted class
        predicted_class = model.predict(
            input_data
        )[0]

        predicted_class_index = list(
            model.classes_
        ).index(predicted_class)

        feature_values = shap_array[
            0,
            :,
            predicted_class_index
        ]

    elif shap_array.ndim == 2:

        feature_values = shap_array[0]

    else:

        raise ValueError(
            f"Unexpected SHAP output shape: "
            f"{shap_array.shape}"
        )

    # --------------------------------------------------------
    # CREATE FEATURE IMPORTANCE TABLE
    # --------------------------------------------------------

    explanation = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "shap_value": feature_values,
        "importance": np.abs(feature_values)
    })

    # Sort by importance
    explanation = explanation.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    return explanation
