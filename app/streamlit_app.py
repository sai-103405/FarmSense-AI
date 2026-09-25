import streamlit as st
import tempfile
from pathlib import Path

from src.prediction.predictor import predict_crop
from src.cnn.predict_disease import predict_disease

from src.api_client import (
    run_agent_api,
    run_disease_agent_api
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FarmSense AI",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌾 FarmSense AI")

st.subheader(
    "AI-Powered Crop Recommendation, Plant Disease Detection "
    "and Agricultural Assistant"
)


st.write(
    """
FarmSense AI combines Machine Learning, Deep Learning,
Retrieval-Augmented Generation and an AI Agent into one
agricultural intelligence platform.
"""
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🌾 Crop Recommendation",
        "🍃 Disease Detection",
        "📚 AI Assistant",
        "🤖 AI Agent"
    ]
)


# ============================================================
# TAB 1 — CROP RECOMMENDATION
# ============================================================

with tab1:

    st.header("🌾 Crop Recommendation")

    st.write(
        """
Enter the soil and environmental conditions of your field.
The Random Forest machine-learning model will recommend
a suitable crop.
"""
    )

    col1, col2 = st.columns(2)

    with col1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=90.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=42.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=43.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=20.9
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=82.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=202.9
        )

    st.divider()

    if st.button(
        "🌱 Recommend Crop",
        key="crop_button"
    ):

        try:

            with st.spinner(
                "Analyzing field conditions..."
            ):

                result = predict_crop(
                    nitrogen=nitrogen,
                    phosphorus=phosphorus,
                    potassium=potassium,
                    temperature=temperature,
                    humidity=humidity,
                    ph=ph,
                    rainfall=rainfall
                )

            st.success(
                f"Recommended Crop: {result['crop']}"
            )

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.subheader(
                "Top Predictions"
            )

            for prediction in result["top_predictions"]:

                st.write(
                    f"**{prediction['crop']}** — "
                    f"{prediction['probability'] * 100:.2f}%"
                )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ============================================================
# TAB 2 — PLANT DISEASE DETECTION
# ============================================================

with tab2:

    st.header("🍃 Plant Disease Detection")

    st.write(
        """
Upload a plant leaf image. The ResNet18 deep-learning
model will predict the most likely disease.
"""
    )

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="disease_upload"
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Leaf",
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Disease",
            key="disease_button"
        ):

            temporary_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(
                        uploaded_file.name
                    ).suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temporary_path = temp_file.name

                with st.spinner(
                    "Analyzing leaf..."
                ):

                    disease, confidence, top_predictions = (
                        predict_disease(
                            temporary_path
                        )
                    )

                st.success(
                    f"Detected Disease: {disease}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.subheader(
                    "Top Predictions"
                )

                for prediction in top_predictions:

                    st.write(
                        f"**{prediction['disease']}** — "
                        f"{prediction['confidence'] * 100:.2f}%"
                    )

                st.info(
                    """
Image-based AI predictions are not definitive diagnoses.
For important agricultural decisions, confirm the condition
with a qualified agricultural expert.
"""
                )

            except Exception as error:

                st.error(
                    f"Disease detection error: {error}"
                )

            finally:

                if temporary_path:

                    try:

                        Path(
                            temporary_path
                        ).unlink(
                            missing_ok=True
                        )

                    except Exception:

                        pass


# ============================================================
# TAB 3 — AI ASSISTANT
# ============================================================

with tab3:

    st.header("📚 FarmSense AI Assistant")

    st.write(
        """
Ask agricultural questions. The assistant uses the
FarmSense knowledge base with Retrieval-Augmented Generation
and the local Llama 3.2 model.
"""
    )

    question = st.text_input(
        "Ask an agricultural question",
        placeholder="Example: What are the symptoms of tomato early blight?",
        key="assistant_question"
    )

    if st.button(
        "💬 Ask FarmSense AI",
        key="assistant_button"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                from src.utils.rag_pipeline import (
                    answer_agriculture_question
                )

                with st.spinner(
                    "Searching agricultural knowledge..."
                ):

                    result = answer_agriculture_question(
                        question
                    )

                st.subheader(
                    "💡 Answer"
                )

                st.write(
                    result["answer"]
                )

                if result.get("sources"):

                    st.subheader(
                        "📚 Retrieved Knowledge"
                    )

                    for source in result["sources"]:

                        if "document" in source:

                            st.write(
                                source["document"]
                            )

            except Exception as error:

                st.error(
                    f"AI Assistant error: {error}"
                )


# ============================================================
# TAB 4 — AI AGENT
# ============================================================

with tab4:

    st.header("🤖 FarmSense AI Agent")

    st.write(
        """
The FarmSense AI Agent automatically decides which
AI capability should handle your request.
"""
    )

    st.info(
        """
🌾 Crop Recommendation → Random Forest

🍃 Disease Detection → ResNet18 CNN

📚 Agricultural Questions → RAG + Llama 3.2
"""
    )

    # ========================================================
    # USER QUESTION
    # ========================================================

    agent_question = st.text_input(
        "What would you like FarmSense AI to do?",
        placeholder="Example: What crop should I grow?",
        key="agent_question"
    )

    st.divider()

    # ========================================================
    # CROP CONDITIONS
    # ========================================================

    st.subheader(
        "🌾 Crop Conditions"
    )

    st.caption(
        """
These values are used when the Agent routes your request
to crop recommendation.
"""
    )

    crop_col1, crop_col2 = st.columns(2)

    with crop_col1:

        agent_nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=90.0,
            key="agent_nitrogen"
        )

        agent_phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=42.0,
            key="agent_phosphorus"
        )

        agent_potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=43.0,
            key="agent_potassium"
        )

        agent_temperature = st.number_input(
            "Temperature (°C)",
            value=20.9,
            key="agent_temperature"
        )

    with crop_col2:

        agent_humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=82.0,
            key="agent_humidity"
        )

        agent_ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            key="agent_ph"
        )

        agent_rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=202.9,
            key="agent_rainfall"
        )

    st.divider()

    # ========================================================
    # DISEASE IMAGE
    # ========================================================

    st.subheader(
        "🍃 Disease Detection"
    )

    st.caption(
        """
Upload a leaf image if your request is about plant
disease detection.
"""
    )

    agent_image = st.file_uploader(
        "Upload a leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="agent_image"
    )

    if agent_image is not None:

        st.image(
            agent_image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

    st.divider()

    # ========================================================
    # RUN AGENT
    # ========================================================

    if st.button(
        "🤖 Run FarmSense AI Agent",
        key="agent_button"
    ):

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not agent_question.strip():

            st.warning(
                "Please enter a request for the Agent."
            )

        else:

            # ------------------------------------------------
            # Prepare crop data
            # ------------------------------------------------

            crop_data = {

                "nitrogen": agent_nitrogen,

                "phosphorus": agent_phosphorus,

                "potassium": agent_potassium,

                "temperature": agent_temperature,

                "humidity": agent_humidity,

                "ph": agent_ph,

                "rainfall": agent_rainfall
            }

            try:

                # ==================================================
                # DISEASE REQUEST
                # ==================================================

                if agent_image is not None:

                    temporary_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=Path(
                                agent_image.name
                            ).suffix
                        ) as temp_file:

                            temp_file.write(
                                agent_image.getbuffer()
                            )

                            temporary_path = temp_file.name

                        with st.spinner(
                            "🤖 Agent is analyzing the leaf..."
                        ):

                            result = run_disease_agent_api(
                                agent_question.strip(),
                                temporary_path
                            )

                    finally:

                        if temporary_path:

                            try:

                                Path(
                                    temporary_path
                                ).unlink(
                                    missing_ok=True
                                )

                            except Exception:

                                pass

                # ==================================================
                # NORMAL AGENT REQUEST
                # ==================================================

                else:

                    with st.spinner(
                        "🤖 Agent is processing your request..."
                    ):

                        result = run_agent_api(
                            agent_question.strip(),
                            crop_data=crop_data
                        )

                # ==================================================
                # AGENT DECISION
                # ==================================================

                st.subheader(
                    "🧭 Agent Decision"
                )

                route = result.get(
                    "route",
                    "unknown"
                )

                st.success(
                    f"Selected capability: {route.upper()}"
                )

                # ==================================================
                # AGENT ANSWER
                # ==================================================

                st.subheader(
                    "💡 Agent Answer"
                )

                st.write(
                    result.get(
                        "answer",
                        "No answer returned."
                    )
                )

                # ==================================================
                # SUPPORTING INFORMATION
                # ==================================================

                sources = result.get(
                    "sources",
                    []
                )

                if sources:

                    st.subheader(
                        "📊 Supporting Information"
                    )

                    for source in sources:

                        # ------------------------------------------
                        # Crop source
                        # ------------------------------------------

                        if (
                            "crop" in source
                            and "probability" in source
                        ):

                            probability = (
                                source["probability"] * 100
                            )

                            st.write(
                                f"**{source['crop']}** — "
                                f"{probability:.2f}%"
                            )

                        # ------------------------------------------
                        # Disease source
                        # ------------------------------------------

                        elif (
                            "disease" in source
                            and "confidence" in source
                        ):

                            confidence = (
                                source["confidence"] * 100
                            )

                            st.write(
                                f"**{source['disease']}** — "
                                f"{confidence:.2f}%"
                            )

                        # ------------------------------------------
                        # RAG source
                        # ------------------------------------------

                        elif "document" in source:

                            st.write(
                                source["document"]
                            )

            # ====================================================
            # ERROR HANDLING
            # ====================================================

            except Exception as error:

                st.error(
                    f"Agent API error: {error}"
                )
