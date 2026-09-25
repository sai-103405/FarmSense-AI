from src.utils.rag_pipeline import answer_agriculture_question
from src.prediction.predictor import predict_crop
from src.cnn.predict_disease import predict_disease


# ============================================================
# ROUTING
# ============================================================

def route_question(question, image_path=None):
    """
    Decide which FarmSense AI capability should handle
    the user's request.

    Routes:
        crop     -> Crop recommendation ML model
        disease  -> Plant disease CNN
        knowledge -> RAG + Llama 3.2
    """

    question_lower = question.lower()

    # --------------------------------------------------------
    # Crop recommendation keywords
    # --------------------------------------------------------

    crop_keywords = [
        "crop",
        "grow",
        "soil",
        "nitrogen",
        "phosphorus",
        "potassium",
        "rainfall",
        "humidity",
        "ph"
    ]

    # --------------------------------------------------------
    # Disease detection keywords
    #
    # These are specifically about detecting a disease
    # from an uploaded image.
    # --------------------------------------------------------

    disease_detection_keywords = [
        "detect disease",
        "detect the disease",
        "identify disease",
        "identify the disease",
        "diagnose this",
        "what disease is this",
        "what disease does this leaf have",
        "what disease is on this leaf",
        "disease in this image",
        "disease on this leaf",
        "disease from this image",
        "analyze this leaf"
    ]

    # --------------------------------------------------------
    # Disease detection requires an image
    # --------------------------------------------------------

    if image_path is not None:

        return "disease"

    # --------------------------------------------------------
    # Crop recommendation
    # --------------------------------------------------------

    if any(
        keyword in question_lower
        for keyword in crop_keywords
    ):

        return "crop"

    # --------------------------------------------------------
    # Text-only disease questions
    #
    # Example:
    # "What are the symptoms of tomato early blight?"
    #
    # These should go to RAG, not CNN detection.
    # --------------------------------------------------------

    if any(
        keyword in question_lower
        for keyword in disease_detection_keywords
    ):

        return "knowledge"

    # --------------------------------------------------------
    # Default route
    # --------------------------------------------------------

    return "knowledge"


# ============================================================
# MAIN AGENT
# ============================================================

def run_agent(
    question,
    crop_data=None,
    image_path=None
):
    """
    Main FarmSense AI Agent.

    Parameters
    ----------
    question : str
        User's request.

    crop_data : dict, optional
        Crop conditions required for crop recommendation.

    image_path : str, optional
        Path to uploaded leaf image for disease detection.

    Returns
    -------
    dict
        Route, answer, and supporting sources/predictions.
    """

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question or not question.strip():

        return {
            "route": "knowledge",
            "answer": "Please enter a question or request.",
            "sources": []
        }

    # --------------------------------------------------------
    # Determine route
    # --------------------------------------------------------

    route = route_question(
        question,
        image_path=image_path
    )

    # ========================================================
    # KNOWLEDGE / RAG
    # ========================================================

    if route == "knowledge":

        result = answer_agriculture_question(
            question,
            top_k=3
        )

        return {
            "route": "knowledge",
            "answer": result["answer"],
            "sources": result["sources"]
        }

    # ========================================================
    # CROP RECOMMENDATION
    # ========================================================

    if route == "crop":

        # Crop data is required
        if crop_data is None:

            return {
                "route": "crop",
                "answer": (
                    "Crop conditions are required "
                    "for crop recommendation."
                ),
                "sources": []
            }

        # Make sure all required fields exist
        required_fields = [
            "nitrogen",
            "phosphorus",
            "potassium",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in crop_data
        ]

        if missing_fields:

            return {
                "route": "crop",
                "answer": (
                    "Missing crop data: "
                    + ", ".join(missing_fields)
                ),
                "sources": []
            }

        # Run crop recommendation model
        result = predict_crop(
            crop_data["nitrogen"],
            crop_data["phosphorus"],
            crop_data["potassium"],
            crop_data["temperature"],
            crop_data["humidity"],
            crop_data["ph"],
            crop_data["rainfall"]
        )

        answer = (
            f"Recommended crop: {result['crop']}\n\n"
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )

        return {
            "route": "crop",
            "answer": answer,
            "sources": result["top_predictions"]
        }

    # ========================================================
    # DISEASE DETECTION
    # ========================================================

    if route == "disease":

        # Image is required
        if image_path is None:

            return {
                "route": "disease",
                "answer": (
                    "A leaf image is required "
                    "for disease detection."
                ),
                "sources": []
            }

        # Run CNN prediction
        disease, confidence, top_predictions = (
            predict_disease(image_path)
        )

        answer = (
            f"Detected disease: {disease}\n\n"
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        return {
            "route": "disease",
            "answer": answer,
            "sources": top_predictions
        }

    # ========================================================
    # FALLBACK
    # ========================================================

    return {
        "route": "knowledge",
        "answer": (
            "FarmSense AI could not determine "
            "which capability should handle "
            "this request."
        ),
        "sources": []
    }


# ============================================================
# TERMINAL TEST MODE
# ============================================================

if __name__ == "__main__":

    print()
    print("================================")
    print("     FARMSENSE AI AGENT")
    print("================================")
    print()

    question = input(
        "Ask FarmSense AI: "
    ).strip()

    result = run_agent(question)

    print()
    print("================================")
    print("AGENT RESULT")
    print("================================")

    print()
    print(
        f"Selected capability: "
        f"{result['route'].upper()}"
    )

    print()
    print("Answer:")
    print(result["answer"])

    if result["sources"]:

        print()
        print("Supporting information:")

        for source in result["sources"]:

            print(source)
