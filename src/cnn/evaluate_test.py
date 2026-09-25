import torch
from sklearn.metrics import accuracy_score, classification_report

from data_pipeline_final import test_loader
from resnet_model import PlantDiseaseResNet


# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# Load model
model = PlantDiseaseResNet(num_classes=38)

model.load_state_dict(
    torch.load(
        "models/cnn/plant_disease_resnet18_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# Store predictions
all_predictions = []
all_labels = []


# Evaluate on completely unseen test set
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# Calculate accuracy
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Detailed classification report
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4
    )
)