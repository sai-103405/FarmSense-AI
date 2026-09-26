import streamlit as st
import tempfile
import textwrap
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
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background: linear-gradient(
            135deg,
            #f7faf7 0%,
            #eef7f0 50%,
            #f8fbf8 100%
        );
    }

    /* Header */

    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #14532d,
            #166534,
            #15803d
        );
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(20, 83, 45, 0.18);
    }

    .hero h1 {
        font-size: 2.8rem;
        margin-bottom: 0.3rem;
        font-weight: 750;
    }

    .hero p {
        font-size: 1.05rem;
        margin: 0.2rem 0;
        opacity: 0.92;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        font-size: 0.82rem;
        margin-top: 0.8rem;
    }

    /* Cards */

    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid #dce9df;
        min-height: 145px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }

    .feature-card h3 {
        margin-bottom: 0.35rem;
    }

    .feature-card p {
        color: #526158;
        font-size: 0.92rem;
    }

    /* Result cards */

    .result-card {
        background: white;
        border-radius: 18px;
        padding: 1.4rem;
        border: 1px solid #d9e8dc;
        margin-top: 1rem;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
    }

    .result-title {
        color: #166534;
        font-size: 1rem;
        font-weight: 650;
    }

    .result-value {
        color: #14532d;
        font-size: 2rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }

    /* Section titles */

    .section-title {
        color: #14532d;
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }

    .section-description {
        color: #5b665f;
        margin-bottom: 1rem;
    }

    /* Small labels */

    .mini-label {
        color: #6b756e;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 650;
    }

    /* Footer */

    .footer {
        margin-top: 3rem;
        padding: 1.5rem;
        text-align: center;
        color: #69746d;
        font-size: 0.82rem;
        border-top: 1px solid #dce7df;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #f1f7f2;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:0.8rem 0 1.2rem 0;
        ">
            <div style="font-size:3rem;">🌱</div>
            <h2 style="margin:0;color:#14532d;">
                FarmSense AI
            </h2>
            <p style="color:#657168;">
                Agricultural Intelligence Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🧠 AI Capabilities")

    st.markdown(
        """
        **🌾 Crop Recommendation**  
        Random Forest ML model

        **🍃 Disease Detection**  
        Fine-tuned ResNet18 CNN

        **📚 Knowledge Assistant**  
        RAG + Llama 3.2

        **🤖 AI Agent**  
        Intelligent capability routing
        """
    )

    st.divider()

    st.markdown("### 📊 Model Highlights")

    st.metric(
        "Crop model accuracy",
        "99.55%"
    )

    st.metric(
        "Disease test accuracy",
        "95.79%"
    )

    st.divider()

    st.caption(
        "FarmSense AI is an experimental decision-support "
        "platform. AI predictions should be verified with "
        "qualified agricultural professionals."
    )


# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">🌱 MULTIMODAL AGRICULTURAL AI</div>'
    '<h1>FarmSense AI</h1>'
    '<p>Intelligent crop recommendations, plant disease detection and agricultural knowledge assistance.</p>'
    '<p>Machine Learning • Deep Learning • RAG • AI Agents</p>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PLATFORM OVERVIEW
# ============================================================

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

with overview_col1:

    st.markdown(
        """
        <div class="feature-card">

        <h3>🌾 Crop Intelligence</h3>

        <p>
        Analyze soil and environmental conditions to generate
        crop recommendations using machine learning.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with overview_col2:

    st.markdown(
        """
        <div class="feature-card">

        <h3>🍃 Vision AI</h3>

        <p>
        Analyze plant leaf images using a fine-tuned ResNet18
        disease classification model.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with overview_col3:

    st.markdown(
        """
        <div class="feature-card">

        <h3>📚 Knowledge AI</h3>

        <p>
        Retrieve relevant agricultural knowledge and generate
        grounded responses using RAG.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with overview_col4:

    st.markdown(
        """
        <div class="feature-card">

        <h3>🤖 AI Agent</h3>

        <p>
        Automatically route requests to the appropriate
        FarmSense AI capability.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


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

    st.markdown(
        '<div class="section-title">🌾 Crop Recommendation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Enter soil and environmental conditions. FarmSense AI
        uses a trained Random Forest model to estimate suitable
        crops.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "💡 Tip: Use measurements from your soil test and local "
        "weather conditions when available."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="mini-label">Soil nutrients</div>',
            unsafe_allow_html=True
        )

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=90.0,
            help="Nitrogen concentration in the soil."
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=42.0,
            help="Phosphorus concentration in the soil."
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=43.0,
            help="Potassium concentration in the soil."
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=20.9
        )

    with col2:

        st.markdown(
            '<div class="mini-label">Environmental conditions</div>',
            unsafe_allow_html=True
        )

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
        "🌱 Analyze Field & Recommend Crop",
        key="crop_button",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing soil and environmental conditions..."
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

            st.markdown(
                """
                <div class="result-card">
                    <div class="result-title">
                        Recommended Crop
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-title">🌱 Recommended crop</div>'
                    f'<div class="result-value">{result["crop"].title()}</div>'
                    '</div>',
                    unsafe_allow_html=True
                )  

            with result_col2:
                
                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-title">🎯 Model confidence</div>'
                    f'<div class="result-value">{result["confidence"] * 100:.2f}%</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.subheader("📊 Top Predictions")

            for prediction in result["top_predictions"]:

                probability = prediction["probability"] * 100

                st.write(
                    f"**{prediction['crop'].title()}**"
                )

                st.progress(
                    min(probability / 100, 1.0)
                )

                st.caption(
                    f"{probability:.2f}% model probability"
                )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ============================================================
# TAB 2 — PLANT DISEASE DETECTION
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">🍃 Plant Disease Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Upload a plant leaf image and let the fine-tuned ResNet18
        vision model identify the most likely disease class.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📤 Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        key="disease_upload"
    )

    if uploaded_file is not None:

        image_col, info_col = st.columns([1.4, 1])

        with image_col:

            st.image(
                uploaded_file,
                caption="Uploaded leaf",
                use_container_width=True
            )

        with info_col:

            st.markdown(
                """
                <div class="result-card">

                <div class="result-title">
                    🔬 Analysis
                </div>

                <p>
                The image will be processed by the FarmSense
                ResNet18 plant-disease classifier.
                </p>

                <p>
                <b>Supported:</b> JPG, JPEG, PNG
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button(
            "🔍 Analyze Leaf",
            key="disease_button",
            use_container_width=True
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
                    "🧠 ResNet18 is analyzing the leaf..."
                ):

                    disease, confidence, top_predictions = (
                        predict_disease(
                            temporary_path
                        )
                    )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <div class="result-title">
                            🍃 Detected class
                        </div>

                        <div class="result-value">
                            {disease.replace("___", " — ").replace("_", " ").title()}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with result_col2:

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <div class="result-title">
                            🎯 Confidence
                        </div>

                        <div class="result-value">
                            {confidence * 100:.2f}%
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.subheader("📊 Top Predictions")

                for prediction in top_predictions:

                    probability = prediction["confidence"] * 100

                    disease_name = (
                        prediction["disease"]
                        .replace("___", " — ")
                        .replace("_", " ")
                        .title()
                    )

                    st.write(
                        f"**{disease_name}**"
                    )

                    st.progress(
                        min(probability / 100, 1.0)
                    )

                    st.caption(
                        f"{probability:.2f}% model confidence"
                    )

                st.warning(
                    """
                    ⚠️ AI image predictions are not definitive
                    diagnoses. For important agricultural decisions,
                    confirm the condition with a qualified
                    agricultural expert.
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

    st.markdown(
        '<div class="section-title">📚 FarmSense AI Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Ask agricultural questions. The assistant retrieves
        relevant knowledge from the FarmSense knowledge base
        and generates a grounded response using local Llama 3.2.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "🧠 Architecture: Sentence Transformers → ChromaDB → "
        "retrieved knowledge → Llama 3.2"
    )

    question = st.text_input(
        "Your agricultural question",
        placeholder=(
            "Example: What are the symptoms of tomato early blight?"
        ),
        key="assistant_question"
    )

    if st.button(
        "💬 Ask FarmSense AI",
        key="assistant_button",
        use_container_width=True
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
                    "📚 Retrieving knowledge and generating answer..."
                ):

                    result = answer_agriculture_question(
                        question
                    )

                st.markdown(
                    """
                    <div class="result-card">
                    <div class="result-title">
                    💡 FarmSense AI Answer
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    result["answer"]
                )

                if result.get("sources"):

                    with st.expander(
                        "📚 View Retrieved Knowledge",
                        expanded=False
                    ):

                        for index, source in enumerate(
                            result["sources"],
                            start=1
                        ):

                            if "document" in source:

                                st.markdown(
                                    f"**Source {index}**"
                                )

                                st.write(
                                    source["document"]
                                )

                                if index < len(
                                    result["sources"]
                                ):
                                    st.divider()

            except Exception as error:

                st.error(
                    f"AI Assistant error: {error}"
                )


# ============================================================
# TAB 4 — AI AGENT
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">🤖 FarmSense AI Agent</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Ask FarmSense to perform an agricultural task. The Agent
        automatically routes your request to the appropriate
        AI capability.
        </div>
        """,
        unsafe_allow_html=True
    )

    agent_info_col1, agent_info_col2, agent_info_col3 = st.columns(3)

    with agent_info_col1:

        st.info(
            """
            🌾 **Crop**

            Random Forest
            crop recommendation
            """
        )

    with agent_info_col2:

        st.info(
            """
            🍃 **Disease**

            ResNet18 image
            classification
            """
        )

    with agent_info_col3:

        st.info(
            """
            📚 **Knowledge**

            RAG + Llama 3.2
            agricultural assistant
            """
        )

    agent_question = st.text_input(
        "What would you like FarmSense AI to do?",
        placeholder=(
            "Example: What crop should I grow with these soil conditions?"
        ),
        key="agent_question"
    )

    st.divider()

    # ========================================================
    # CROP CONDITIONS
    # ========================================================

    st.subheader("🌾 Crop Conditions")

    st.caption(
        "These values are supplied to the Agent when it routes "
        "your request to crop recommendation."
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

    st.subheader("🍃 Disease Detection")

    st.caption(
        "Upload a leaf image when your request requires "
        "image-based disease detection."
    )

    agent_image = st.file_uploader(
        "📤 Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        key="agent_image"
    )

    if agent_image is not None:

        st.image(
            agent_image,
            caption="Agent input leaf image",
            width=400
        )

    st.divider()

    # ========================================================
    # RUN AGENT
    # ========================================================

    if st.button(
        "🤖 Run FarmSense AI Agent",
        key="agent_button",
        use_container_width=True
    ):

        if not agent_question.strip():

            st.warning(
                "Please enter a request for the Agent."
            )

        else:

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

                route = result.get(
                    "route",
                    "unknown"
                )

                route_display = {
                    "crop": "🌾 Crop Recommendation",
                    "disease": "🍃 Disease Detection",
                    "knowledge": "📚 Agricultural Knowledge"
                }.get(
                    route.lower(),
                    route.upper()
                )

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="result-title">
                        🧭 Agent Decision
                    </div>

                    <div class="result-value">
                        {route_display}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ==================================================
                # AGENT ANSWER
                # ==================================================

                st.markdown(
                    """
                    <div class="result-card">

                    <div class="result-title">
                        💡 Agent Answer
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
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

                    with st.expander(
                        "📊 View Supporting Information",
                        expanded=True
                    ):

                        for source in sources:

                            if (
                                "crop" in source
                                and "probability" in source
                            ):

                                probability = (
                                    source["probability"] * 100
                                )

                                st.write(
                                    f"**{source['crop'].title()}**"
                                )

                                st.progress(
                                    min(probability / 100, 1.0)
                                )

                                st.caption(
                                    f"{probability:.2f}% probability"
                                )

                            elif (
                                "disease" in source
                                and "confidence" in source
                            ):

                                confidence = (
                                    source["confidence"] * 100
                                )

                                disease_name = (
                                    source["disease"]
                                    .replace("___", " — ")
                                    .replace("_", " ")
                                    .title()
                                )

                                st.write(
                                    f"**{disease_name}**"
                                )

                                st.progress(
                                    min(confidence / 100, 1.0)
                                )

                                st.caption(
                                    f"{confidence:.2f}% confidence"
                                )

                            elif "document" in source:
                                st.write(
                                    source["document"]
                                )

            except Exception:
                st.error(
                    "⚠️ Agent temporarily unavailable. "
                    "Please try again in a moment."
                )


# ============================================================
# FOOTER
# ============================================================


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<b>🌱 FarmSense AI</b><br>'
    'Production-oriented multimodal agricultural AI platform<br><br>'
    'Machine Learning • Deep Learning • RAG • AI Agents • FastAPI • Streamlit • Docker'
    '<br><br>'
    '⚠️ AI-generated predictions are decision-support outputs '
    'and should not replace qualified agricultural advice.',
    unsafe_allow_html=True
)
