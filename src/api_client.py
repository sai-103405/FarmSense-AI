import requests


# ============================================================
# FASTAPI SERVER
# ============================================================

API_URL = "http://host.docker.internal:8000"


# ============================================================
# CHECK API
# ============================================================

def check_api():
    """
    Check whether the FarmSense AI FastAPI server is running.
    """

    response = requests.get(
        f"{API_URL}/",
        timeout=10
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# CROP RECOMMENDATION API
# ============================================================

def predict_crop_api(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):
    """
    Send crop conditions to the FastAPI crop prediction endpoint.
    """

    payload = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# AI AGENT API
# ============================================================

def run_agent_api(
    question,
    crop_data=None
):
    """
    Send a question to the FarmSense AI Agent.

    Optionally sends crop conditions when
    crop recommendation is required.
    """

    payload = {
        "question": question
    }

    if crop_data is not None:

        payload.update({
            "nitrogen": crop_data["nitrogen"],
            "phosphorus": crop_data["phosphorus"],
            "potassium": crop_data["potassium"],
            "temperature": crop_data["temperature"],
            "humidity": crop_data["humidity"],
            "ph": crop_data["ph"],
            "rainfall": crop_data["rainfall"]
        })

    response = requests.post(
        f"{API_URL}/agent",
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# DISEASE AGENT API
# ============================================================

def run_disease_agent_api(
    question,
    image_path
):
    """
    Send a leaf image to the FastAPI disease-agent endpoint.
    """

    with open(image_path, "rb") as image_file:

        files = {
            "image": image_file
        }

        data = {
            "question": question
        }

        response = requests.post(
            f"{API_URL}/agent/disease",
            data=data,
            files=files,
            timeout=180
        )

    response.raise_for_status()

    return response.json()
