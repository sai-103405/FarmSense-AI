import joblib
from pathlib import Path


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the trained model
MODEL_PATH = PROJECT_ROOT / "models" / "crop_recommendation_rf.pkl"


def load_model():
    """
    Load and return the trained crop recommendation model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model