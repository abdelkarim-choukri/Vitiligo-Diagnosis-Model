import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2  # robust fallback reader

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def load_rgb(path: str) -> Image.Image:
    """Robust image loader: try Pillow; fallback to OpenCV if Pillow fails."""
    try:
        img = Image.open(path)
        img.load()  # force decode early
        return img.convert('RGB')
    except Exception:
        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Unusable image: {path}")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)


def load_gray(path: str) -> Image.Image:
    """Robust grayscale loader for masks (if/when you use them)."""
    try:
        img = Image.open(path)
        img.load()
        return img.convert('L')
    except Exception:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Unusable mask: {path}")
        return Image.fromarray(arr)


class VitiligoDataset(Dataset):
    """
    Dataset for paired clinical and Wood's lamp images with optional masks.

    CSV columns required: ['clinical_filename', 'wood_filename', 'label']
    Optional: 'mask_filename' (used only if mask_dir is provided)
    """

    def __init__(self, csv_file, clinical_dir, wood_dir, mask_dir=None, transform=None, image_size=224):
        self.df = pd.read_csv(csv_file)
        self.clinical_dir = clinical_dir
        self.wood_dir = wood_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # torchvision transforms
        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build absolute paths
        cli_path = os.path.join(self.clinical_dir, str(row['clinical_filename']).strip())
        woo_path = os.path.join(self.wood_dir,     str(row['wood_filename']).strip())
        label = int(row['label'])

        # Robust load
        cli_img = load_rgb(cli_path)
        woo_img = load_rgb(woo_path)

        # Optional mask
        mask = None
        if self.mask_dir and ('mask_filename' in self.df.columns) and pd.notna(row.get('mask_filename', None)):
            mask_path = os.path.join(self.mask_dir, str(row['mask_filename']).strip())
            mask = load_gray(mask_path)

        # Joint albumentations (if provided): expects additional_targets={"image2":"image","mask":"mask"}
        if self.transform is not None:
            data = {'image': np.array(cli_img), 'image2': np.array(woo_img)}
            if mask is not None:
                data['mask'] = np.array(mask)
            aug = self.transform(**data)
            cli_img = Image.fromarray(aug['image'])
            woo_img = Image.fromarray(aug['image2'])
            if mask is not None:
                mask = Image.fromarray(aug['mask'])

        # Resize + to tensor + normalize
        cli_img = self.resize(cli_img)
        woo_img = self.resize(woo_img)
        cli_tensor = self.normalize(self.to_tensor(cli_img))
        woo_tensor = self.normalize(self.to_tensor(woo_img))

        # Mask tensor (placeholder if None)
        if mask is not None:
            mask = self.resize(mask)
            mask_arr = (np.array(mask).astype('float32') / 255.0)
            mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # [1,H,W]
        else:
            mask_tensor = torch.tensor(0.0)  # scalar placeholder; unused when seg is off

        # Use float label to match BCEWithLogitsLoss input
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return cli_tensor, woo_tensor, mask_tensor, label_tensor
