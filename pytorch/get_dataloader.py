import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
from sklearn.model_selection import train_test_split

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.model_selection import train_test_split


def get_auto_transforms(model_name, pretrained=True):
    """Get automatic transforms with augmentations for any torchvision model"""
    weights = None
    if pretrained:
        try:  # Handle MobileNetV2 and other models uniformly
            weights = getattr(models, f"{model_name.upper()}_Weights").DEFAULT
        except AttributeError:
            weights = None

    # Get native transforms from weights
    if weights:
        base_transform = weights.transforms()
        # Extract parameters for augmentation
        size = weights.meta['size'][0] if 'size' in weights.meta else 224
        crop_pct = weights.meta.get('crop_pct', 0.875)
        resize_size = int(math.ceil(size / crop_pct))

        # Enhanced training augmentations
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(weights.meta['mean'], weights.meta['std'])
        ])

        val_tf = base_transform  # Use native validation transform

    else:  # Fallback for custom models
        size = 224
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return train_tf, val_tf

# Rest of the code remains the same with these IMPROVEMENTS:

# 1. Fixed image extension check in scan_folder:


def scan_folder(root):
    paths, labels, class_to_idx = [], [], {}
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')  # Added common extensions
    for idx, cls in enumerate(sorted(os.listdir(root))):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        class_to_idx[cls] = idx
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(valid_exts):  # Fixed extension check
                paths.append(os.path.join(cls_dir, fn))
                labels.append(idx)
    return paths, labels, class_to_idx

# 2. Enhanced dataset class with error handling:


class FaceExpressionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            return self[(idx + 1) % len(self)]  # Skip bad images


def get_dataloaders(data_path,
                    batch_size=32,
                    num_workers=4,
                    test_split=0.2,
                    seed=42,
                    model_name='mobilenet_v2',
                    auto_transform=True):
    all_paths, all_labels, class_to_idx = scan_folder(data_path)

    # 2) split into train / test, then train / val
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=test_split/(1-test_split),
        random_state=seed,
        stratify=all_labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=test_split/(1-test_split),
        random_state=seed,
        stratify=train_labels
    )
    if auto_transform:
        train_tf, val_tf = get_auto_transforms(model_name)
    else:
        train_tf = val_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    train_ds = FaceExpressionDataset(train_paths, train_labels, train_tf)
    val_ds = FaceExpressionDataset(val_paths,   val_labels,   val_tf)
    test_ds = FaceExpressionDataset(test_paths,  test_labels,   val_tf)
    train_loader = DataLoader(train_ds,   batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds,     batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds,    batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Loaded {len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
          f"samples across {len(class_to_idx)} classes.")
    return train_loader, val_loader, test_loader, class_to_idx
