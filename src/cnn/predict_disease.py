import torch
from PIL import Image
from torchvision import transforms

from src.cnn.resnet_model import PlantDiseaseResNet
from src.cnn.class_names import class_mapping


# ============================================================
# SELECT DEVICE
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = PlantDiseaseResNet(num_classes=38)

model.load_state_dict(
    torch.load(
        "models/cnn/plant_disease_resnet18_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================================
# PREDICT PLANT DISEASE
# ============================================================

def predict_disease(image_path):
    """
    Predict the top 3 possible plant diseases
    from a leaf image.
    """

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    # --------------------------------------------------------
    # GET TOP 3 PREDICTIONS
    # --------------------------------------------------------

    top_probabilities, top_classes = torch.topk(
        probabilities,
        k=3,
        dim=1
    )

    top_predictions = []

    for probability, class_id in zip(
        top_probabilities[0],
        top_classes[0]
    ):

        class_id = class_id.item()
        confidence = probability.item()

        disease_name = class_mapping[class_id]

        top_predictions.append({
            "disease": disease_name,
            "confidence": confidence
        })

    # --------------------------------------------------------
    # MAIN PREDICTION
    # --------------------------------------------------------

    predicted_disease = top_predictions[0]["disease"]
    predicted_confidence = top_predictions[0]["confidence"]

    return (
        predicted_disease,
        predicted_confidence,
        top_predictions
    )


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "Enter the path to a leaf image: "
    ).strip()

    disease, confidence, top_predictions = predict_disease(
        image_path
    )

    print("\n==============================")
    print("PLANT DISEASE PREDICTION")
    print("==============================")

    print(
        f"\nTop Prediction: {disease}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("\nTop 3 Predictions:")

    for index, prediction in enumerate(
        top_predictions,
        start=1
    ):

        print(
            f"{index}. "
            f"{prediction['disease']} — "
            f"{prediction['confidence'] * 100:.2f}%"
        )
