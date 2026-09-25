from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import tempfile
from pathlib import Path

from src.prediction.predictor import predict_crop
from src.agent import run_agent


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FarmSense AI",
    description="AI-powered agricultural AI platform",
    version="1.0.0"
)


# ============================================================
# CROP DATA MODEL
# ============================================================

class FarmData(BaseModel):

    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# ============================================================
# AGENT REQUEST MODEL
# ============================================================

class AgentRequest(BaseModel):

    question: str

    # Crop information
    nitrogen: float | None = None
    phosphorus: float | None = None
    potassium: float | None = None
    temperature: float | None = None
    humidity: float | None = None
    ph: float | None = None
    rainfall: float | None = None


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "FarmSense AI API is running!"
    }


# ============================================================
# CROP PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: FarmData):

    try:

        result = predict_crop(
            nitrogen=data.nitrogen,
            phosphorus=data.phosphorus,
            potassium=data.potassium,
            temperature=data.temperature,
            humidity=data.humidity,
            ph=data.ph,
            rainfall=data.rainfall
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# AI AGENT ENDPOINT
# ============================================================

@app.post("/agent")
def agent(data: AgentRequest):

    try:

        # ----------------------------------------------------
        # Create crop data only if supplied
        # ----------------------------------------------------

        crop_data = None

        crop_values = [
            data.nitrogen,
            data.phosphorus,
            data.potassium,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall
        ]

        # If all crop values are provided,
        # create crop_data dictionary.

        if all(value is not None for value in crop_values):

            crop_data = {
                "nitrogen": data.nitrogen,
                "phosphorus": data.phosphorus,
                "potassium": data.potassium,
                "temperature": data.temperature,
                "humidity": data.humidity,
                "ph": data.ph,
                "rainfall": data.rainfall
            }

        # ----------------------------------------------------
        # Run FarmSense AI Agent
        # ----------------------------------------------------

        result = run_agent(
            question=data.question,
            crop_data=crop_data
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# DISEASE DETECTION ENDPOINT
# ============================================================

@app.post("/agent/disease")
async def agent_disease(
    question: str = "What disease is on this leaf?",
    image: UploadFile = File(...)
):

    temporary_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded image temporarily
        # ----------------------------------------------------

        suffix = Path(image.filename).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            contents = await image.read()

            temp_file.write(contents)

            temporary_path = temp_file.name

        # ----------------------------------------------------
        # Run Agent with image
        # ----------------------------------------------------

        result = run_agent(
            question=question,
            image_path=temporary_path
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary image
        # ----------------------------------------------------

        if temporary_path:

            try:

                Path(temporary_path).unlink(
                    missing_ok=True
                )

            except Exception:

                pass
