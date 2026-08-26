"""
PyTorch Dataset + DataLoaders for watermark forensics.
"""

import cv2
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WatermarkForensicsDataset(Dataset):
    def __init__(self, split_csv: Path, img_size: int = 224, train: bool = False):
        self.df = pd.read_csv(split_csv)
        self.img_size = img_size
        self.train = train

        if train:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["path"]
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(
                f"Could not read image: {img_path}"
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return img, label


def create_dataloaders(
    splits_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    torch.manual_seed(seed)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    for csv_file in [train_csv, val_csv, test_csv]:
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Missing split file: {csv_file}\n"
                f"Expected train.csv, val.csv and test.csv "
                f"inside {splits_dir}"
            )

    train_ds = WatermarkForensicsDataset(train_csv, img_size, train=True)
    val_ds = WatermarkForensicsDataset(val_csv, img_size, train=False)
    test_ds = WatermarkForensicsDataset(test_csv, img_size, train=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl