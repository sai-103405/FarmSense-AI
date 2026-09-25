import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_pipeline_final import test_loader
from resnet_model import PlantDiseaseResNet
from class_names import class_mapping


# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# Load fine-tuned ResNet18 model
model = PlantDiseaseResNet(num_classes=38)

model.load_state_dict(
    torch.load(
        "models/cnn/plant_disease_resnet18_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# Store predictions and true labels
all_predictions = []
all_labels = []


# Run prediction on the unseen test set
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# Create confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(38))
)


# Create class names in the correct order
class_names = [
    class_mapping[i]
    for i in range(38)
]


# Plot confusion matrix
plt.figure(figsize=(18, 15))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title(
    "Plant Disease Classification - Test Set Confusion Matrix"
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()


# Save the confusion matrix
output_path = (
    "models/cnn/plant_disease_test_confusion_matrix.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved successfully!")
print(output_path)


# Display the plot
plt.show()