import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class VitiligoDataset(Dataset):
    """
    PyTorch Dataset for paired clinical and Wood's lamp images with optional segmentation masks.

    Expects a CSV with columns: ['clinical_filename', 'wood_filename', 'label']
    Optionally, if mask_dir is provided and the CSV has 'mask_filename', it will load masks.

    Args:
        csv_file (str): Path to the CSV file.
        clinical_dir (str): Directory containing clinical images.
        wood_dir (str): Directory containing Wood's lamp images.
        mask_dir (str, optional): Directory containing mask images. Defaults to None.
        transform (albumentations.Compose, optional): Joint augmentation for images/mask. Defaults to None.
        image_size (int, optional): Resize images/masks to this size. Defaults to 224.
    """
    def __init__(self, csv_file, clinical_dir, wood_dir, mask_dir=None, transform=None, image_size=224):
        self.df = pd.read_csv(csv_file)
        self.clinical_dir = clinical_dir
        self.wood_dir = wood_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # torchvision transforms for resizing, tensor conversion, and normalization
        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Build file paths
        cli_path = os.path.join(self.clinical_dir, row['clinical_filename'])
        woo_path = os.path.join(self.wood_dir, row['wood_filename'])
        label = int(row['label'])

        # Load images
        cli_img = Image.open(cli_path).convert('RGB')
        woo_img = Image.open(woo_path).convert('RGB')

        # Load mask if available
        mask = None
        if self.mask_dir and 'mask_filename' in row and pd.notna(row['mask_filename']):
            mask_path = os.path.join(self.mask_dir, row['mask_filename'])
            mask = Image.open(mask_path).convert('L')

        # Apply joint augmentations if provided (expects albumentations)
        if self.transform:
            # Prepare dict for albumentations
            data = {'image': np.array(cli_img), 'image2': np.array(woo_img)}
            if mask is not None:
                data['mask'] = np.array(mask)
            augmented = self.transform(**data)
            cli_img = Image.fromarray(augmented['image'])
            woo_img = Image.fromarray(augmented['image2'])
            if mask is not None:
                mask = Image.fromarray(augmented['mask'])

        # Resize, to tensor, normalize
        cli_img = self.resize(cli_img)
        woo_img = self.resize(woo_img)
        cli_tensor = self.normalize(self.to_tensor(cli_img))
        woo_tensor = self.normalize(self.to_tensor(woo_img))

        # Process mask to tensor
        if mask is not None:
            mask = self.resize(mask)
            mask_arr = np.array(mask).astype('float32') / 255.0
            mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        else:
            mask_tensor = torch.tensor(0.)  # placeholder

        label_tensor = torch.tensor(label, dtype=torch.long)
        return cli_tensor, woo_tensor, mask_tensor, label_tensor
