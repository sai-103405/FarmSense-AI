import torch
import torch.nn as nn
from torchvision import models


class PlantDiseaseResNet(nn.Module):

    def __init__(self, num_classes=38):

        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Get number of inputs to the final layer
        num_features = self.model.fc.in_features

        # Replace final layer for our 38 disease classes
        self.model.fc = nn.Linear(
            num_features,
            num_classes
        )

        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Allow only the new classification layer to learn
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):

        return self.model(x)


if __name__ == "__main__":

    model = PlantDiseaseResNet(num_classes=38)

    sample = torch.randn(1, 3, 224, 224)

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
