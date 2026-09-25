import pandas as pd
from src.models.model_loader import load_model
from src.utils.validators import validate_field_data


def predict_crop(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):
    """
    Predict the most suitable crop based on
    soil and weather conditions.
    """

    validate_field_data(
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall
    )

    model = load_model()

    input_data = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    probability_df = pd.DataFrame({
        "crop": classes,
        "probability": probabilities
    }).sort_values(
        "probability",
        ascending=False
    )

    top_probability = probability_df.iloc[0]["probability"]

    return {
        "crop": prediction,
        "confidence": float(top_probability),
        "top_predictions": probability_df.head(5).to_dict(
            orient="records"
        )
    }