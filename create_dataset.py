import os

base_path = "dataset/train"
folders = ["normal", "tumor"]

for folder in folders:
    path = os.path.join(base_path, folder)
    os.makedirs(path, exist_ok=True)

print("Dataset folders created successfully!")