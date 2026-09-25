from datasets import load_from_disk
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch


DATASET_PATH = "data/raw/PlantVillage/tiny"


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


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


dataset = load_from_disk(DATASET_PATH)

indices = list(range(len(dataset)))
labels = dataset["class_idx"]

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


train_dataset = PlantVillageDataset(
    dataset,
    train_indices,
    transform=train_transform
)

val_dataset = PlantVillageDataset(
    dataset,
    val_indices,
    transform=val_transform
)


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


if __name__ == "__main__":

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    print("Total images:", len(dataset))
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Number of classes:", len(set(labels)))

    print(
        "Training batch shape:",
        train_images.shape
    )

    print(
        "Validation batch shape:",
        val_images.shape
    )