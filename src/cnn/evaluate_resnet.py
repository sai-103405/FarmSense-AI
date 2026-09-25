import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data_pipeline import val_loader
from resnet_model import PlantDiseaseResNet


# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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


all_labels = []
all_predictions = []


# Run validation data through model
with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# Accuracy
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

accuracy = (all_labels == all_predictions).mean() * 100

print(f"\nValidation Accuracy: {accuracy:.2f}%")


# Classification report
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4
    )
)


# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)
