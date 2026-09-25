import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from datasets import load_from_disk

from data_pipeline import val_loader
from resnet_model import PlantDiseaseResNet


# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# Load dataset
dataset = load_from_disk("data/raw/PlantVillage/tiny")


# Build class names
class_mapping = {}

for item in dataset:
    class_idx = item["class_idx"]
    host = item["host"]
    disease = item["disease"]

    class_mapping[class_idx] = f"{host} - {disease}"


class_names = [
    class_mapping[i]
    for i in sorted(class_mapping)
]


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


# Predictions
with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )


# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)


# Plot
plt.figure(figsize=(18, 16))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("FarmSense AI - Plant Disease Confusion Matrix")
plt.xlabel("Predicted Disease")
plt.ylabel("Actual Disease")

plt.xticks(
    rotation=90,
    fontsize=7
)

plt.yticks(
    rotation=0,
    fontsize=7
)

plt.tight_layout()


# Save
plt.savefig(
    "models/cnn/plant_disease_confusion_matrix.png",
    dpi=200
)

plt.show()

print(
    "\nConfusion matrix saved successfully!"
)
