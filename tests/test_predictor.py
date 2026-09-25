from src.prediction.predictor import predict_crop


def test_crop_prediction():
    result = predict_crop(
        nitrogen=90,
        phosphorus=42,
        potassium=43,
        temperature=20.9,
        humidity=82,
        ph=6.5,
        rainfall=202.9
    )

    assert "crop" in result
    assert "confidence" in result
    assert "top_predictions" in result

    assert isinstance(result["crop"], str)
    assert result["crop"] != ""

    assert 0 <= result["confidence"] <= 1

    assert isinstance(
        result["top_predictions"],
        list
    )

    assert len(result["top_predictions"]) == 5


def test_crop_prediction_top_result():
    result = predict_crop(
        nitrogen=90,
        phosphorus=42,
        potassium=43,
        temperature=20.9,
        humidity=82,
        ph=6.5,
        rainfall=202.9
    )

    top_prediction = result["top_predictions"][0]

    assert "crop" in top_prediction
    assert "probability" in top_prediction

    assert (
        top_prediction["probability"]
        >= 0
    )

    assert (
        top_prediction["probability"]
        <= 1
    )
