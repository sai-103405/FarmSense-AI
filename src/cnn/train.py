import torch
import torch.nn as nn
import torch.optim as optim

from data_pipeline import train_loader, val_loader
from model import PlantDiseaseCNN


# Use M1 GPU if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# Model
model = PlantDiseaseCNN(num_classes=38)
model = model.to(device)


# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training settings
epochs = 5


for epoch in range(epochs):

    # -------------------------
    # Training
    # -------------------------

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


    # -------------------------
    # Validation
    # -------------------------

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


# Save model
torch.save(
    model.state_dict(),
    "models/cnn/plant_disease_cnn.pth"
)

print("Model saved successfully!")