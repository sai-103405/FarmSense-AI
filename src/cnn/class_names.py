from datasets import load_from_disk

DATASET_PATH = "data/raw/PlantVillage/tiny"

dataset = load_from_disk(DATASET_PATH)

class_mapping = {}

for item in dataset:
    class_idx = item["class_idx"]
    host = item["host"]
    disease = item["disease"]

    class_mapping[class_idx] = f"{host} - {disease}"


print("\nPlant Disease Classes:\n")

for class_idx in sorted(class_mapping):
    print(f"Class {class_idx}: {class_mapping[class_idx]}")
