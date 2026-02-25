# !/usr/bin/env python3
"""
Selective retraining pipeline for problematic dog breeds.

This script fine-tunes a lightweight classifier on a subset of target breeds
that show lower precision in the main model, while avoiding full retraining
of the complete breed taxonomy.
"""

import os
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class SelectiveBreedDataset(Dataset):
    """Dataset that loads samples only from selected target breeds."""

    def __init__(self, data_dir, transform=None, target_breeds=None):
        """
Initialize selective dataset.

Args:
data_dir: Base directory containing per-breed subdirectories.
transform: Optional torchvision transform pipeline.
target_breeds: List of breed directory names to include.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_breeds = target_breeds or []

        self.samples = []
        self.class_to_idx = {}

        available_breeds = [
            directory
            for directory in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, directory)) and directory in self.target_breeds
        ]

        for idx, breed in enumerate(available_breeds):
            self.class_to_idx[breed] = idx
            breed_path = os.path.join(data_dir, breed)

            for image_file in os.listdir(breed_path):
                if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(breed_path, image_file), idx))

        print(f"Dataset created: {len(self.samples)} images, {len(available_breeds)} classes")
        for breed, idx in self.class_to_idx.items():
            count = sum(1 for sample in self.samples if sample[1] == idx)
            print(f"  {idx}: {breed} - {count} images")

    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load one sample and apply transforms, with robust fallback on IO errors."""
        image_path, label = self.samples[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as exc:
            print(f"Error loading {image_path}: {exc}")
            return self.__getitem__((idx + 1) % len(self.samples))


class SelectiveBreedClassifier(nn.Module):
    """ResNet34-based classifier for selective fine-tuning."""

    def __init__(self, num_classes):
        """Create model head with target class count."""
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, inputs):
        """Forward pass."""
        return self.backbone(inputs)


def create_selective_fine_tuning():
    """Run selective fine-tuning on predefined low-performing breeds."""
    print("Starting selective retraining for problematic breeds")
    print("=" * 70)

    target_breeds = [
        "Labrador_retriever",
        "Norwegian_elkhound",
        "beagle",
        "pug",
        "basset",
        "Samoyed",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SelectiveBreedDataset(
        "breed_processed_data/train",
        transform=train_transform,
        target_breeds=target_breeds,
    )

    val_dir = "breed_processed_data/val"
    if not os.path.exists(val_dir):
        print("Creating validation dataset split...")
        os.makedirs(val_dir, exist_ok=True)

        for breed in target_breeds:
            breed_train = f"breed_processed_data/train/{breed}"
            breed_val = f"{val_dir}/{breed}"

            if os.path.exists(breed_train):
                os.makedirs(breed_val, exist_ok=True)

                images = [
                    file_name
                    for file_name in os.listdir(breed_train)
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                val_count = max(1, len(images) // 5)
                val_images = random.sample(images, val_count)

                for image_name in val_images:
                    src = os.path.join(breed_train, image_name)
                    dst = os.path.join(breed_val, image_name)
                    shutil.move(src, dst)

                print(f"  {breed}: moved {len(val_images)} images to validation")

    val_dataset = SelectiveBreedDataset(
        val_dir,
        transform=val_transform,
        target_breeds=target_breeds,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.class_to_idx)
    model = SelectiveBreedClassifier(num_classes).to(device)

    pretrained_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    if os.path.exists(pretrained_path):
        print("Loading pretrained checkpoint as initialization...")
        checkpoint = torch.load(pretrained_path, map_location=device)

        pretrained_dict = checkpoint["model_state_dict"]
        model_dict = model.state_dict()

        pretrained_dict = {
            key: value
            for key, value in pretrained_dict.items()
            if key in model_dict and "fc" not in key
        }

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Backbone loaded. Fine-tuning classifier head.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"\nStarting fine-tuning ({num_classes} classes)...")

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(20):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/20, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch + 1}/20:")
        print(f"  Train: Loss {train_loss / len(train_loader):.4f}, Acc {train_acc:.2f}%")
        print(f"  Val:   Loss {val_loss / len(val_loader):.4f}, Acc {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            os.makedirs("selective_models", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "class_to_idx": train_dataset.class_to_idx,
                    "target_breeds": target_breeds,
                },
                "selective_models/best_selective_model.pth",
            )

            print(f"New best model saved: {val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print("Early stopping triggered")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return "selective_models/best_selective_model.pth", train_dataset.class_to_idx


if __name__ == "__main__":
    model_path, class_mapping = create_selective_fine_tuning()
    print(f"Selective model saved: {model_path}")
    print(f"Classes: {list(class_mapping.keys())}")
