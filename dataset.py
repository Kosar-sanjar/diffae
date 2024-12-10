# dataset.py

import lmdb
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from typing import Optional, Callable, Any, List, Tuple, Dict
from torchvision import transforms
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_transform(image_size: int, flip_prob: float = 0.5, crop_d2c: bool = False) -> Callable:
    """
    Create a torchvision transform pipeline for image preprocessing.

    Args:
        image_size (int): Desired image size after transformation.
        flip_prob (float): Probability of horizontal flipping.
        crop_d2c (bool): Whether to apply a specific cropping strategy (e.g., D2C crop).

    Returns:
        Callable: Composed torchvision transforms.
    """
    transform_list = []
    if crop_d2c:
        # Example placeholder for a specific cropping method
        # Replace with actual D2C cropping logic if necessary
        transform_list.append(transforms.CenterCrop(image_size))
    else:
        transform_list.append(transforms.Resize(image_size))
    
    transform_list.extend([
        transforms.RandomHorizontalFlip(flip_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize to [-1, 1]
    ])
    return transforms.Compose(transform_list)


class BaseDataset(Dataset):
    """
    An abstract base dataset class that other datasets can inherit from.
    """

    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class LMDBDataset(BaseDataset):
    """
    A generic dataset for loading data from an LMDB database.
    """

    def __init__(self, lmdb_path: str, transform: Optional[Callable] = None):
        """
        Initialize the LMDBDataset.

        Args:
            lmdb_path (str): Path to the LMDB database.
            transform (Optional[Callable]): Transformations to apply to the images.
        """
        super().__init__(transform)
        self.lmdb_path = lmdb_path
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB path {lmdb_path} does not exist.")
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(b'length').decode())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieve the data at index idx.

        Returns:
            Dict: A dictionary containing the data fields.
        """
        key = f"data-{str(idx).zfill(5)}".encode("utf-8")
        with self.env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            data = pickle.loads(data_bytes)
        
        # For 'eeg_encoder.lmdb', data contains only 'data' (EEG)
        # For 'eeg_ddim_ffhq.lmdb', data contains 'data' (EEG), 'image', 'label', 'subject'
        if 'image' in data:
            # EEG + Image Dataset (Conditional DDIM)
            eeg = torch.tensor(data['data'], dtype=torch.float32)
            image = data['image']  # Serialized PIL Image or bytes
            label = data.get('label', None)
            subject = data.get('subject', None)
            
            # If image is stored as bytes, convert it back to PIL Image
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image)).convert('RGB')
            elif not isinstance(image, Image.Image):
                # If image is stored as a NumPy array or other format
                image = Image.fromarray(image).convert('RGB')
            else:
                # Ensure image is in RGB format
                image = image.convert('RGB')
            
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
            
            return {
                'eeg': eeg,
                'image': image,
                'label': label,
                'subject': subject
            }
        else:
            # EEG Encoder Dataset (EEG only)
            eeg = torch.tensor(data['data'], dtype=torch.float32)
            return {
                'eeg': eeg
            }


class EEGEncoderDataset(Dataset):
    """
    Dataset class for training the Semantic Encoder with EEG data only.
    """

    def __init__(self, encoder_lmdb_path: str, transform: Optional[Callable] = None):
        """
        Initialize the EEGEncoderDataset.

        Args:
            encoder_lmdb_path (str): Path to the 'eeg_encoder.lmdb' database.
            transform (Optional[Callable]): Transformations to apply (unused here).
        """
        super().__init__(transform)
        self.lmdb_path = encoder_lmdb_path
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB path {self.lmdb_path} does not exist.")
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(b'length').decode())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieve the EEG data at index idx.

        Returns:
            Dict: A dictionary containing the EEG data.
        """
        key = f"data-{str(idx).zfill(5)}".encode("utf-8")
        with self.env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            eeg_data = pickle.loads(data_bytes)  # This is 'eeg_ma' as a NumPy array

        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        return {
            'eeg': eeg_tensor
        }


class EEGImageDDIMDataset(Dataset):
    """
    Dataset class for training the Conditional DDIM with EEG and Image data.
    """

    def __init__(self, ddim_lmdb_path: str, transform: Optional[Callable] = None):
        """
        Initialize the EEGImageDDIMDataset.

        Args:
            ddim_lmdb_path (str): Path to the 'eeg_ddim_ffhq.lmdb' database.
            transform (Optional[Callable]): Transformations to apply to the images.
        """
        super().__init__(transform)
        self.lmdb_path = ddim_lmdb_path
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB path {self.lmdb_path} does not exist.")
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(b'length').decode())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieve the EEG and Image data at index idx.

        Returns:
            Dict: A dictionary containing the EEG data, image, label, and subject.
        """
        key = f"data-{str(idx).zfill(5)}".encode("utf-8")
        with self.env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            data = pickle.loads(data_bytes)
        
        # Extract EEG data
        eeg_np = data['data']
        eeg_tensor = torch.tensor(eeg_np, dtype=torch.float32)
        
        # Extract and process image
        image = data['image']
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            # If image is stored as a NumPy array or other format
            image = Image.fromarray(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extract label and subject
        label = data.get('label', None)
        subject = data.get('subject', None)
        
        return {
            'eeg': eeg_tensor,
            'image': image,
            'label': label,
            'subject': subject
        }


# Example Usage
if __name__ == "__main__":
    # Paths to LMDB databases
    encoder_lmdb_path = 'datasets/eeg_encoder.lmdb'
    ddim_lmdb_path = 'datasets/eeg_ddim_ffhq.lmdb'
    
    # Define image transformations
    transform = make_transform(image_size=128, flip_prob=0.5)
    
    # Initialize Datasets
    eeg_encoder_dataset = EEGEncoderDataset(
        encoder_lmdb_path=encoder_lmdb_path,
        transform=None  # No transformation for EEG data
    )
    
    ddim_dataset = EEGImageDDIMDataset(
        ddim_lmdb_path=ddim_lmdb_path,
        transform=transform
    )
    
    # Initialize DataLoaders
    eeg_encoder_loader = DataLoader(
        eeg_encoder_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    ddim_loader = DataLoader(
        ddim_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Iterate through the DataLoaders
    for batch in eeg_encoder_loader:
        eeg = batch['eeg']  # Tensor of shape (batch_size, eeg_dim)
        # Proceed with training the semantic encoder
        print(f"EEG Encoder Batch - EEG Shape: {eeg.shape}")
        break  # Remove this break in actual training
    
    for batch in ddim_loader:
        eeg = batch['eeg']          # Tensor of shape (batch_size, eeg_dim)
        images = batch['image']     # Tensor of shape (batch_size, 3, image_size, image_size)
        labels = batch.get('label') # Depends on your data structure
        subjects = batch.get('subject') # Depends on your data structure
        # Proceed with training the Conditional DDIM
        print(f"DDIM Batch - EEG Shape: {eeg.shape}, Images Shape: {images.shape}")
        break  # Remove this break in actual training
