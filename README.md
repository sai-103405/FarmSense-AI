# 🌱 FarmSense AI

### Production-Grade Multimodal AI Decision Platform for Agriculture

FarmSense AI is an end-to-end artificial intelligence platform designed to support agricultural decision-making through **machine learning, computer vision, explainable AI, retrieval-augmented generation (RAG), and AI agents**.

The platform combines crop recommendation, plant disease detection, agricultural knowledge retrieval, and an AI agent into a unified application exposed through **FastAPI, Streamlit, and Docker**.

---

## 🚀 Key Features

### 🌾 1. AI Crop Recommendation

Recommends suitable crops based on agricultural and environmental conditions.

**Input features:**

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

**Model:**

- Random Forest Classifier
- 200 estimators
- Stratified train/test split
- Test accuracy: **99.55%**

The model also provides the top predicted crops with their probability scores.

---

### 🍃 2. Plant Disease Detection

Uses a fine-tuned **ResNet18 convolutional neural network** to identify plant diseases from leaf images.

The model was trained using a PlantVillage-based dataset containing:

- 38 disease/healthy classes
- RGB leaf images
- 224 × 224 model input

**Model progression:**

| Model | Validation Accuracy |
|---|---:|
| Custom CNN | 55.79% |
| ResNet18 | 82.89% |
| Fine-tuned ResNet18 | **94.47%** |

Final held-out test performance:

**Test Accuracy: 95.79%**

Additional test metrics:

- Macro Precision: 0.9605
- Macro Recall: 0.9572
- Macro F1 Score: 0.9567

---

## 🔍 Explainable AI

FarmSense AI includes explainability components to make model predictions easier to understand.

### SHAP

SHAP is used to analyze feature importance for the crop recommendation model.

This helps identify how agricultural variables contribute to the model's predictions.

### Grad-CAM

Grad-CAM is implemented for the plant disease CNN to visualize image regions that contribute to the model's prediction.

This provides a visual explanation of the disease classification.

---

## 🤖 Retrieval-Augmented Generation (RAG)

FarmSense AI includes a local agricultural knowledge assistant.

The RAG pipeline:

1. Loads agricultural knowledge documents
2. Splits documents into knowledge chunks
3. Generates embeddings using Sentence Transformers
4. Stores embeddings in ChromaDB
5. Retrieves relevant knowledge for a user question
6. Sends retrieved context to a local LLM
7. Generates a grounded response

### Local LLM

The project uses:

**Ollama + Llama 3.2 3B**

This allows the RAG assistant to run locally without requiring a paid external LLM API.

The system prompt instructs the model to use retrieved knowledge, avoid unsupported claims, and treat image-based disease predictions as non-definitive.

---

## 🧠 AI Agent

FarmSense AI includes an AI agent that routes user requests to the appropriate capability.

### Agent routing

```text
                    User Question
                         │
                         ▼
                  ┌──────────────┐
                  │   AI Agent   │
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Crop Request    Disease Image   Knowledge
          │              │              │
          ▼              ▼              ▼
     Random Forest    ResNet18        RAG
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    Final Response
```

The agent can route requests to:

- Crop recommendation
- Plant disease detection
- Agricultural knowledge retrieval

---

## 🏗️ System Architecture

```text
                         ┌─────────────────┐
                         │     User        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Streamlit UI  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     FastAPI     │
                         └────────┬────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
                 ▼                ▼                ▼
        ┌────────────────┐ ┌──────────────┐ ┌───────────────┐
        │ Crop Model     │ │ Disease CNN  │ │   AI Agent    │
        │ Random Forest  │ │   ResNet18   │ │               │
        └────────────────┘ └──────────────┘ └───────┬───────┘
                                                     │
                                             ┌───────▼───────┐
                                             │ RAG Pipeline   │
                                             └───────┬───────┘
                                                     │
                                             ┌───────▼───────┐
                                             │ ChromaDB       │
                                             │ + Ollama       │
                                             └───────────────┘
```

---

## 🛠️ Technology Stack

### Programming

- Python 3.11

### Machine Learning

- Scikit-learn
- Random Forest
- Logistic Regression
- Gradient Boosting
- Decision Trees

### Deep Learning

- PyTorch
- TorchVision
- ResNet18
- CNN
- Grad-CAM

### Explainable AI

- SHAP

### Generative AI

- Ollama
- Llama 3.2 3B
- Retrieval-Augmented Generation (RAG)

### Vector Search

- ChromaDB
- Sentence Transformers

### Backend

- FastAPI
- Pydantic

### Frontend

- Streamlit

### Testing

- PyTest

### Deployment

- Docker
- Docker Desktop

---

## 📂 Project Structure

```text
FarmSense-AI/
│
├── app/
│   ├── main.py
│   └── streamlit_app.py
│
├── data/
│   └── knowledge_base/
│       └── plant_disease_guide.txt
│
├── models/
│   ├── cnn/
│   │   └── plant_disease_resnet18_finetuned.pth
│   │
│   └── crop_recommendation_rf.pkl
│
├── notebooks/
│   └── 01_farm_data_exploration.ipynb
│
├── src/
│   ├── cnn/
│   │   ├── class_names.py
│   │   ├── create_split.py
│   │   ├── data_pipeline.py
│   │   ├── data_pipeline_final.py
│   │   ├── evaluate_resnet.py
│   │   ├── evaluate_test.py
│   │   ├── fine_tune_resnet.py
│   │   ├── gradcam.py
│   │   ├── model.py
│   │   ├── predict_disease.py
│   │   ├── resnet_model.py
│   │   ├── train.py
│   │   └── train_resnet.py
│   │
│   ├── models/
│   │   └── model_loader.py
│   │
│   ├── prediction/
│   │   ├── crop_explainer.py
│   │   └── predictor.py
│   │
│   ├── utils/
│   │   ├── build_vector_db.py
│   │   ├── rag_pipeline.py
│   │   ├── retriever.py
│   │   └── validators.py
│   │
│   ├── agent.py
│   └── api_client.py
│
├── tests/
│   ├── test_agent.py
│   ├── test_predictor.py
│   └── test_validators.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pytest.ini
├── requirements.txt
└── requirements-prod.txt
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FarmSense-AI
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

For development:

```bash
pip install -r requirements.txt
```

For production:

```bash
pip install -r requirements-prod.txt
```

---

## 🧠 Build the Vector Database

Run:

```bash
python src/utils/build_vector_db.py
```

This creates the ChromaDB vector database from the agricultural knowledge base.

---

## 🤖 Run Ollama

Install Ollama and download the model:

```bash
ollama pull llama3.2:3b
```

Start Ollama:

```bash
ollama serve
```

---

## 🚀 Start FastAPI

Run:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🎨 Start Streamlit

Run:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit application will normally be available at:

```text
http://localhost:8501
```

---

## 🐳 Docker Deployment

Build the production image:

```bash
docker build -t farmsense-ai:optimized .
```

Run the container:

```bash
docker run -d \
  --name farmsense-ai-optimized \
  -p 8503:8501 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  farmsense-ai:optimized
```

Open:

```text
http://localhost:8503
```

The Docker deployment builds the agricultural vector database during the image build.

---

## 🧪 Testing

Run the test suite:

```bash
pytest
```

The project includes tests for:

- Input validation
- Crop prediction
- AI agent routing

---

## 📊 Model Performance

### Crop Recommendation

**Random Forest**

- Test Accuracy: **99.55%**
- Number of estimators: 200
- Dataset size: 2,200 samples
- Number of crop classes: 22

### Plant Disease Detection

**Fine-tuned ResNet18**

- Validation Accuracy: **94.47%**
- Held-out Test Accuracy: **95.79%**
- Classes: 38
- Macro Precision: **0.9605**
- Macro Recall: **0.9572**
- Macro F1: **0.9567**

---

## 🔐 Responsible AI

FarmSense AI is designed as a decision-support and educational system.

Model predictions should not be treated as definitive agricultural diagnoses.

For plant disease predictions, users should verify results using appropriate agricultural experts, local agricultural extension services, or other trusted sources.

The RAG assistant is also instructed to avoid unsupported claims and unsafe chemical or pesticide recommendations.

---

## 🔮 Future Improvements

Potential future development includes:

- Larger and more diverse agricultural datasets
- Additional crop and disease classes
- Cloud deployment
- Authentication and user accounts
- Production monitoring
- Model versioning
- Automated model retraining
- Advanced agent tool calling
- Weather API integration
- Soil and satellite-data integration
- Mobile application
- More comprehensive agricultural knowledge sources

---

## 👨‍💻 Project Focus

FarmSense AI demonstrates practical implementation of:

- Machine Learning
- Deep Learning
- Computer Vision
- Explainable AI
- Generative AI
- Retrieval-Augmented Generation
- AI Agents
- REST APIs
- Automated Testing
- Containerization
- Local LLM deployment

The project is designed as a portfolio demonstration of building an AI system from **data processing and model development through API integration, RAG, agent routing, testing, and Docker deployment**.
