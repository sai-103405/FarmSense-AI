from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import torch

DATASET_PATH = "data/raw/PlantVillage/tiny"

dataset = load_from_disk(DATASET_PATH)

indices = list(range(len(dataset)))
labels = dataset["class_idx"]

# 70% training, 30% temporary
train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# Split remaining 30% into:
# 15% validation + 15% test
temp_labels = [labels[i] for i in temp_indices]

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

# Save the three splits
torch.save(train_indices, "data/processed/train_indices.pt")
torch.save(val_indices, "data/processed/val_indices.pt")
torch.save(test_indices, "data/processed/test_indices.pt")

print("Dataset split created successfully!")
print("Total images:", len(dataset))
print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))
print("Test images:", len(test_indices))

print("\nFiles saved:")
print("data/processed/train_indices.pt")
print("data/processed/val_indices.pt")
print("data/processed/test_indices.pt")
