import numpy as np
import cv2
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import timm
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def scan_folder(root):
    paths, labels, class_to_idx = [], [], {}
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    class_dirs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    for idx, cls in enumerate(class_dirs):
        cls_dir = os.path.join(root, cls)
        class_to_idx[cls] = idx  # Assign index starting from 0
        print(f"Class {cls} -> index {idx}")
        class_images = 0
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(valid_exts):
                paths.append(os.path.join(cls_dir, fn))
                labels.append(idx)
                class_images += 1
        print(f"  Found {class_images} images for class {cls}")
    print(f"Total images: {len(paths)}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Label range: min={min(labels) if labels else 'N/A'}, max={max(labels) if labels else 'N/A'}")

    return paths, labels, class_to_idx


class FaceExpressionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        if self.labels:
            min_label = min(self.labels)
            max_label = max(self.labels)
            print(f"Dataset label range: min={min_label}, max={max_label}")
            if min_label < 0:
                print("WARNING: Negative labels detected!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            if len(self) > 1:
                alt_idx = (idx + 1) % len(self)
                if alt_idx != idx:
                    return self.__getitem__(alt_idx)
            dummy = torch.zeros(3, 224, 224)
            return dummy, 0


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_clahe = clahe.apply(img_gray)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_clahe = np.stack([img_clahe] * 3, axis=-1)
        return transforms.functional.to_pil_image(img_clahe)


def get_auto_transforms(model_name: str, pretrained: bool = True):
    if model_name == "proxyless_nas":
        train_transforms_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        train_transforms_list.append(CLAHETransform())
        train_transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        train_tf = transforms.Compose(train_transforms_list)
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        model = timm.create_model(model_name, pretrained=pretrained)
        data_cfg = timm.data.resolve_model_data_config(model)
        val_tf = timm.data.create_transform(**data_cfg, is_training=False)
        train_tf = timm.data.create_transform(**data_cfg, is_training=True)
        grayscale_transform = transforms.Grayscale(num_output_channels=3)
        clahe_transform = CLAHETransform()  # Your custom CLAHE transform
        train_tf.transforms.insert(0, grayscale_transform)
        train_tf.transforms.insert(-2, clahe_transform)
        val_tf.transforms.insert(0, grayscale_transform)
        del model
    torch.cuda.empty_cache()
    return train_tf, val_tf
# testing


def get_dataloaders(data_path="./preprocessed/nodup_data_face_only",
                    batch_size=32,
                    num_workers=4,
                    test_split=0.2,
                    seed=42,
                    model_name='mobilenetv2_100',):
    all_paths, all_labels, class_to_idx = scan_folder(data_path)
    num_classes = len(class_to_idx)
    if not all_paths:
        raise ValueError(f"No images found in {data_path}")
    if max(all_labels) >= num_classes:
        raise ValueError(f"Label index {max(all_labels)} exceeds class count {num_classes}")
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=test_split,
        random_state=seed,
        stratify=all_labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=test_split,
        random_state=seed,
        stratify=train_labels
    )
    train_tf, val_tf = get_auto_transforms(model_name)
    train_ds = FaceExpressionDataset(train_paths, train_labels, train_tf)
    val_ds = FaceExpressionDataset(val_paths, val_labels, val_tf)
    test_ds = FaceExpressionDataset(test_paths, test_labels, val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Loaded {len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
          f"samples across {num_classes} classes.")

    return train_loader, val_loader, test_loader, class_to_idx
