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
from torchvision.transforms import InterpolationMode
import math
set_seed(0)


def get_auto_transforms(model_name: str, pretrained: bool = True):
    fn = getattr(models, model_name)
    model = fn(pretrained=pretrained)
    cfg = model.default_cfg
    size = cfg['input_size'][-1]
    mean, std = cfg['mean'], cfg['std']
    interp = InterpolationMode[cfg.get('interpolation', 'bilinear').upper()]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, interpolation=interp),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    crop_pct = cfg.get('crop_pct', 0.875)
    resize_size = math.ceil(size / crop_pct)
    val_tf = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interp),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


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


def get_dataloaders(data_path=None,
                    batch_size=32,
                    num_workers=4,
                    test_split=0.5,
                    seed=42,
                    model_name='mobilenet_v2',
                    auto_transform=True):
    set_seed(seed)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if data_path is None or not os.path.exists(data_path):
        data_path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
        print(f"Dataset downloaded to: {data_path}")
    train_dir = os.path.join(data_path, "images/train")
    val_dir = os.path.join(data_path, "images/validation")
    if auto_transform:
        train_transform, val_test_transform = get_auto_transforms(model_name)
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC,
                              antialias=True),
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
