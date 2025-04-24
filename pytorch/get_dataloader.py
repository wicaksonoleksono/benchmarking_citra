import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub
from sklearn.model_selection import train_test_split
from util import set_seed
# Set random seed for reproducibility

# Custom Dataset class

set_seed(0)


class FaceExpressionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_path=None, batch_size=32, num_workers=4, test_split=0.5, seed=42):
    set_seed(seed)
    if data_path is None or not os.path.exists(data_path):
        data_path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
        print(f"Dataset downloaded to: {data_path}")
    train_dir = os.path.join(data_path, "images/train")
    val_dir = os.path.join(data_path, "images/validation")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_images = []
    train_labels = []
    class_to_idx = {}
    for class_idx, class_name in enumerate(sorted(os.listdir(train_dir))):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            class_to_idx[class_name] = class_idx
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    train_images.append(img_path)
                    train_labels.append(class_idx)
    val_images = []
    val_labels = []

    for class_name in sorted(os.listdir(val_dir)):
        class_dir = os.path.join(val_dir, class_name)
        if os.path.isdir(class_dir):
            class_idx = class_to_idx.get(class_name, len(class_to_idx))
            if class_name not in class_to_idx:
                class_to_idx[class_name] = class_idx

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    val_images.append(img_path)
                    val_labels.append(class_idx)

    val_images_final, test_images, val_labels_final, test_labels = train_test_split(
        val_images, val_labels, test_size=test_split, random_state=seed, stratify=val_labels
    )
    train_dataset = FaceExpressionDataset(train_images, train_labels, transform=train_transform)
    val_dataset = FaceExpressionDataset(val_images_final, val_labels_final, transform=val_test_transform)
    test_dataset = FaceExpressionDataset(test_images, test_labels, transform=val_test_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"Dataset loaded successfully!")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")

    return train_loader, val_loader, test_loader, class_to_idx

# dataloader for testing pipeline
# def get_dataloaders(data_path=None, batch_size=32, num_workers=4, test_split=0.5, seed=42,
#                     limit_train=200, limit_val=50, limit_test=50):
#     set_seed(seed)
#     if data_path is None or not os.path.exists(data_path):
#         data_path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
#         print(f"Dataset downloaded to: {data_path}")

#     train_dir = os.path.join(data_path, "images/train")
#     val_dir = os.path.join(data_path, "images/validation")

#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     val_test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     train_images = []
#     train_labels = []
#     class_to_idx = {}
#     for class_idx, class_name in enumerate(sorted(os.listdir(train_dir))):
#         class_dir = os.path.join(train_dir, class_name)
#         if os.path.isdir(class_dir):
#             class_to_idx[class_name] = class_idx
#             for img_name in os.listdir(class_dir):
#                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(class_dir, img_name)
#                     train_images.append(img_path)
#                     train_labels.append(class_idx)

#     val_images = []
#     val_labels = []

#     for class_name in sorted(os.listdir(val_dir)):
#         class_dir = os.path.join(val_dir, class_name)
#         if os.path.isdir(class_dir):
#             class_idx = class_to_idx.get(class_name, len(class_to_idx))
#             if class_name not in class_to_idx:
#                 class_to_idx[class_name] = class_idx

#             for img_name in os.listdir(class_dir):
#                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(class_dir, img_name)
#                     val_images.append(img_path)
#                     val_labels.append(class_idx)

#     # Make sure to limit the data before the train_test_split to maintain class balance
#     if limit_train > 0 and limit_train < len(train_images):
#         # Stratified sampling to maintain class distribution
#         train_indices = []
#         unique_labels = set(train_labels)
#         samples_per_class = limit_train // len(unique_labels)

#         for label in unique_labels:
#             label_indices = [i for i, l in enumerate(train_labels) if l == label]
#             if len(label_indices) > samples_per_class:
#                 selected_indices = random.sample(label_indices, samples_per_class)
#                 train_indices.extend(selected_indices)
#             else:
#                 train_indices.extend(label_indices)

#         # Limit the train set
#         train_images = [train_images[i] for i in train_indices]
#         train_labels = [train_labels[i] for i in train_indices]
#         print(f"Limited training samples to {len(train_images)}")

#     # Split validation and test
#     val_images_final, test_images, val_labels_final, test_labels = train_test_split(
#         val_images, val_labels, test_size=test_split, random_state=seed, stratify=val_labels
#     )

#     # Limit validation set
#     if limit_val > 0 and limit_val < len(val_images_final):
#         val_indices = []
#         unique_val_labels = set(val_labels_final)
#         samples_per_class = limit_val // len(unique_val_labels)

#         for label in unique_val_labels:
#             label_indices = [i for i, l in enumerate(val_labels_final) if l == label]
#             if len(label_indices) > samples_per_class:
#                 selected_indices = random.sample(label_indices, samples_per_class)
#                 val_indices.extend(selected_indices)
#             else:
#                 val_indices.extend(label_indices)

#         val_images_final = [val_images_final[i] for i in val_indices]
#         val_labels_final = [val_labels_final[i] for i in val_indices]
#         print(f"Limited validation samples to {len(val_images_final)}")

#     # Limit test set
#     if limit_test > 0 and limit_test < len(test_images):
#         test_indices = []
#         unique_test_labels = set(test_labels)
#         samples_per_class = limit_test // len(unique_test_labels)

#         for label in unique_test_labels:
#             label_indices = [i for i, l in enumerate(test_labels) if l == label]
#             if len(label_indices) > samples_per_class:
#                 selected_indices = random.sample(label_indices, samples_per_class)
#                 test_indices.extend(selected_indices)
#             else:
#                 test_indices.extend(label_indices)

#         test_images = [test_images[i] for i in test_indices]
#         test_labels = [test_labels[i] for i in test_indices]
#         print(f"Limited test samples to {len(test_images)}")

#     # Create the datasets with limited data
#     train_dataset = FaceExpressionDataset(train_images, train_labels, transform=train_transform)
#     val_dataset = FaceExpressionDataset(val_images_final, val_labels_final, transform=val_test_transform)
#     test_dataset = FaceExpressionDataset(test_images, test_labels, transform=val_test_transform)

#     # Add class names to datasets for visualization
#     train_dataset.classes = list(class_to_idx.keys())
#     val_dataset.classes = list(class_to_idx.keys())
#     test_dataset.classes = list(class_to_idx.keys())

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     print(f"Dataset loaded successfully!")
#     print(f"Number of training samples: {len(train_dataset)}")
#     print(f"Number of validation samples: {len(val_dataset)}")
#     print(f"Number of test samples: {len(test_dataset)}")
#     print(f"Number of classes: {len(class_to_idx)}")

#     return train_loader, val_loader, test_loader, class_to_idx
