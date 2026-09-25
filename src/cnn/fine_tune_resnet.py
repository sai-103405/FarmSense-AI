import torch
import torch.nn as nn
import torch.optim as optim

from data_pipeline import train_loader, val_loader
from resnet_model import PlantDiseaseResNet


# Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# Create model
model = PlantDiseaseResNet(num_classes=38)

# Load our previously trained ResNet18
model.load_state_dict(
    torch.load(
        "models/cnn/plant_disease_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)


# Unfreeze the last ResNet block
for param in model.model.layer4.parameters():
    param.requires_grad = True

# Keep earlier layers frozen
for param in model.model.layer1.parameters():
    param.requires_grad = False

for param in model.model.layer2.parameters():
    param.requires_grad = False

for param in model.model.layer3.parameters():
    param.requires_grad = False

# Classification layer remains trainable
for param in model.model.fc.parameters():
    param.requires_grad = True


# Loss
criterion = nn.CrossEntropyLoss()


# Small learning rate for fine-tuning
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)


epochs = 5


for epoch in range(epochs):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total


    # Validation
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss / len(val_loader):.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


# Save fine-tuned model
torch.save(
    model.state_dict(),
    "models/cnn/plant_disease_resnet18_finetuned.pth"
)

print("Fine-tuned ResNet18 model saved successfully!")
