"""Dataset CytologIA (recadrage par bbox) + construction des DataLoaders."""

import os
import json
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = ["", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size, train):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def resolve_image_path(images_dir, name):
    """Résout le chemin d'une image en testant plusieurs extensions."""
    direct = os.path.join(images_dir, name)
    if os.path.isfile(direct):
        return direct
    base, ext = os.path.splitext(name)
    if ext:
        return None
    for e in IMG_EXTENSIONS:
        candidate = os.path.join(images_dir, base + e)
        if os.path.isfile(candidate):
            return candidate
    return None


class CytologiaDataset(Dataset):
    def __init__(self, csv_path, images_dir, label_mapping=None, img_size=224,
                 bbox_margin=0.1, train=True, indices=None, verbose=True):
        self.images_dir = images_dir
        self.img_size = img_size
        self.bbox_margin = bbox_margin
        self.train = train

        df = pd.read_csv(csv_path)
        required_cols = {"NAME", "x1", "y1", "x2", "y2", "class"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

        valid = (df["x2"] > df["x1"]) & (df["y2"] > df["y1"])
        n_invalid = (~valid).sum()
        if n_invalid > 0 and verbose:
            warnings.warn(f"{n_invalid} bboxes invalides ignorées.")
        df = df[valid].reset_index(drop=True)

        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        self.df = df

        if label_mapping is None:
            classes = sorted(df["class"].astype(str).unique().tolist())
            label_mapping = {c: i for i, c in enumerate(classes)}
        self.label_mapping = label_mapping
        self.num_classes = len(label_mapping)
        self.transform = get_transforms(img_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row["NAME"])
        x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        label = self.label_mapping[str(row["class"])]

        img_path = resolve_image_path(self.images_dir, name)
        if img_path is None:
            raise FileNotFoundError(f"Image introuvable pour NAME='{name}' dans {self.images_dir}")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        mx, my = (x2 - x1) * self.bbox_margin, (y2 - y1) * self.bbox_margin
        crop = image.crop((max(0, x1 - mx), max(0, y1 - my),
                           min(w, x2 + mx), min(h, y2 + my)))
        return self.transform(crop), label


def make_train_val_datasets(csv_path, images_dir, img_size=224, val_ratio=0.15,
                            bbox_margin=0.1, seed=42, label_mapping_path=None):
    df = pd.read_csv(csv_path)
    valid = (df["x2"] > df["x1"]) & (df["y2"] > df["y1"])
    df = df[valid].reset_index(drop=True)

    classes = sorted(df["class"].astype(str).unique().tolist())
    label_mapping = {c: i for i, c in enumerate(classes)}
    if label_mapping_path:
        os.makedirs(os.path.dirname(label_mapping_path), exist_ok=True)
        with open(label_mapping_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    labels = df["class"].astype(str).map(label_mapping).values
    all_idx = np.arange(len(df))
    try:
        train_idx, val_idx = train_test_split(
            all_idx, test_size=val_ratio, random_state=seed, stratify=labels)
    except ValueError as e:
        warnings.warn(f"Split stratifié impossible ({e}). Repli sur split aléatoire.")
        train_idx, val_idx = train_test_split(all_idx, test_size=val_ratio, random_state=seed)

    kw = dict(label_mapping=label_mapping, img_size=img_size, bbox_margin=bbox_margin)
    train_ds = CytologiaDataset(csv_path, images_dir, train=True, indices=train_idx, **kw)
    val_ds = CytologiaDataset(csv_path, images_dir, train=False, indices=val_idx, **kw)
    return train_ds, val_ds, label_mapping


def make_dataloaders(train_ds, val_ds, batch_size, num_workers):
    """num_workers=0 : sûr sur Windows. num_workers>0 : contexte 'spawn' + workers persistants."""
    common = dict(batch_size=batch_size, pin_memory=True)
    if num_workers == 0:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
        val_loader = DataLoader(val_ds, shuffle=False, **common)
    else:
        wk = dict(num_workers=num_workers, persistent_workers=True,
                  multiprocessing_context="spawn", prefetch_factor=2)
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common, **wk)
        val_loader = DataLoader(val_ds, shuffle=False, **common, **wk)
    return train_loader, val_loader
