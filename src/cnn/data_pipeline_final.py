from datasets import load_from_disk
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch


DATASET_PATH = "data/raw/PlantVillage/tiny"

dataset = load_from_disk(DATASET_PATH)


class PlantVillageDataset(Dataset):

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        real_index = self.indices[index]

        item = self.dataset[real_index]

        image = item["image"]
        label = item["class_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load saved splits
train_indices = torch.load(
    "data/processed/train_indices.pt"
)

val_indices = torch.load(
    "data/processed/val_indices.pt"
)

test_indices = torch.load(
    "data/processed/test_indices.pt"
)


# Create datasets
train_dataset = PlantVillageDataset(
    dataset,
    train_indices,
    transform=train_transform
)

val_dataset = PlantVillageDataset(
    dataset,
    val_indices,
    transform=eval_transform
)

test_dataset = PlantVillageDataset(
    dataset,
    test_indices,
    transform=eval_transform
)


# Create loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


if __name__ == "__main__":

    print("Total images:", len(dataset))
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    train_images, train_labels = next(iter(train_loader))

    print("Training batch shape:", train_images.shape)
    print("Training labels shape:", train_labels.shape)
