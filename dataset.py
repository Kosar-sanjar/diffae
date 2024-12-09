# dataset.py

import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pandas as pd

import torchvision.transforms.functional as Ftrans


# ---------------------------
# Helper Classes
# ---------------------------

class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(img, self.x1, self.y1, self.x2 - self.x1,
                          self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)


def make_transform(
    image_size: int,
    flip_prob: float = 0.5,
    crop_d2c: bool = False,
    do_augment: bool = True,
    do_transform: bool = True,
    do_normalize: bool = True,
) -> Callable:
    """
    Create a transformation pipeline for images.
    """
    transform = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if crop_d2c:
        transform = [
            d2c_crop(),
            transforms.Resize(image_size),
        ]
    if do_augment:
        transform.append(transforms.RandomHorizontalFlip(p=flip_prob))
    if do_transform:
        transform.append(transforms.ToTensor())
    if do_normalize:
        transform.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform)


# ---------------------------
# Base LMDB Dataset
# ---------------------------

class BaseLMDB(Dataset):
    def __init__(self, path: str, original_resolution: int, zfill: int = 5):
        """
        Base class for LMDB-based datasets.

        Args:
            path (str): Path to the LMDB file.
            original_resolution (int): Resolution of the original images.
            zfill (int, optional): Zero-padding for keys. Defaults to 5.
        """
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            length_str = txn.get(b'length')
            if length_str is None:
                raise ValueError("LMDB dataset missing 'length' key.")
            self.length = int(length_str.decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict:
        with self.env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode('utf-8')
            data = txn.get(key)

        if data is None:
            raise IndexError(f"Index {index} not found in LMDB.")

        # Assuming data is serialized as a dictionary with possible keys: 'eeg' and 'image'
        try:
            store_data = pickle.loads(data)
        except Exception as e:
            raise ValueError(f"Failed to deserialize data for index {index}: {e}")

        return store_data


# ---------------------------
# Specific Dataset Classes
# ---------------------------

class ImageDataset(Dataset):
    def __init__(
        self,
        folder: str,
        image_size: int,
        exts: Optional[list] = ['jpg'],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names: bool = False,
        has_subdir: bool = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # relative paths (make it shorter, saves memory and faster to sort)
        if has_subdir:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'**/*.{ext}')
            ]
        else:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'*.{ext}')
            ]
        if sort_names:
            self.paths = sorted(self.paths)

        self.transform = make_transform(
            image_size=image_size,
            flip_prob=0.5,
            crop_d2c=False,
            do_augment=do_augment,
            do_transform=do_transform,
            do_normalize=do_normalize
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict:
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, size: int):
        assert len(dataset) >= size, "Subset size larger than dataset."
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> Dict:
        assert index < self.size, "Index out of bounds for subset."
        return self.dataset[index]


class FFHQlmdb(Dataset):
    def __init__(self,
                 path: str = os.path.expanduser('datasets/ffhq256.lmdb'),
                 image_size: int = 256,
                 original_resolution: int = 256,
                 split: Optional[str] = None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k assuming total length is 70k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == 'test':
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError(f"Split '{split}' is not supported.")

        self.transform = make_transform(
            image_size=image_size,
            do_augment=do_augment,
            do_transform=as_tensor,
            do_normalize=do_normalize
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict:
        assert index < self.length, "Index out of bounds for FFHQlmdb."
        index = index + self.offset
        store_data = self.data[index]
        img = store_data['image']  # Assuming 'image' key exists
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class CelebAlmdb(Dataset):
    """
    Also supports for d2c crop.
    """
    def __init__(self,
                 path: str,
                 image_size: int,
                 original_resolution: int = 128,
                 split: Optional[str] = None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 crop_d2c: bool = False,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)
        self.crop_d2c = crop_d2c

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError(f"Split '{split}' is not supported for CelebAlmdb.")

        if crop_d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(image_size),
            ]
        else:
            transform = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict:
        assert index < self.length, "Index out of bounds for CelebAlmdb."
        index = index + self.offset
        store_data = self.data[index]
        img = store_data['image']  # Assuming 'image' key exists
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Horse_lmdb(Dataset):
    def __init__(self,
                 path: str = os.path.expanduser('datasets/horse256.lmdb'),
                 image_size: int = 128,
                 original_resolution: int = 256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        self.transform = make_transform(
            image_size=image_size,
            do_augment=do_augment,
            do_transform=do_transform,
            do_normalize=do_normalize
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict:
        img = self.data[index]
        img = img['image']  # Assuming 'image' key exists
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Bedroom_lmdb(Dataset):
    def __init__(self,
                 path: str = os.path.expanduser('datasets/bedroom256.lmdb'),
                 image_size: int = 128,
                 original_resolution: int = 256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        self.transform = make_transform(
            image_size=image_size,
            do_augment=do_augment,
            do_transform=do_transform,
            do_normalize=do_normalize
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict:
        img = self.data[index]
        img = img['image']  # Assuming 'image' key exists
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class CelebAttrDataset(Dataset):

    id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 folder: str,
                 image_size: int = 64,
                 attr_path: str = os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext: str = 'png',
                 only_cls_name: Optional[str] = None,
                 only_cls_value: Optional[int] = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 crop_d2c: bool = False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.ext = ext

        # relative paths (make it shorter, saves memory and faster to sort)
        paths = [
            str(p.relative_to(folder))
            for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]
        paths = [str(each).split('.')[0] + '.jpg' for each in paths]

        self.transform = make_transform(
            image_size=image_size,
            flip_prob=0.5,
            crop_d2c=crop_d2c,
            do_augment=do_augment,
            do_transform=do_transform,
            do_normalize=do_normalize
        )

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)
            self.df = self.df[self.df.index.isin(paths)]

        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

    def pos_count(self, cls_name: str) -> int:
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name: str) -> int:
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.folder, name)
        img = Image.open(path).convert('RGB')

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebD2CAttrDataset(CelebAttrDataset):
    """
    The dataset is used in the D2C paper.
    It has a specific crop from the original CelebA.
    """
    def __init__(self,
                 folder: str,
                 image_size: int = 64,
                 attr_path: str = os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext: str = 'jpg',
                 only_cls_name: Optional[str] = None,
                 only_cls_value: Optional[int] = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__(folder,
                         image_size=image_size,
                         attr_path=attr_path,
                         ext=ext,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         crop_d2c=True)


class CelebAttrFewshotDataset(Dataset):

    def __init__(
        self,
        cls_name: str,
        K: int,
        img_folder: str,
        img_size: int = 64,
        ext: str = 'png',
        seed: int = 0,
        only_cls_name: Optional[str] = None,
        only_cls_value: Optional[int] = None,
        all_neg: bool = False,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        d2c: bool = False,
    ):
        self.cls_name = cls_name
        self.K = K
        self.img_folder = img_folder
        self.ext = ext

        if all_neg:
            path = f'data/celeba_fewshots/K{K}_allneg_{cls_name}_{seed}.csv'
        else:
            path = f'data/celeba_fewshots/K{K}_{cls_name}_{seed}.csv'
        self.df = pd.read_csv(path, index_col=0)
        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

        if d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(img_size),
            ]
        else:
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
            ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def pos_count(self, cls_name: str) -> int:
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name: str) -> int:
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, _ = img_name.split('.')
        img_path = os.path.join(self.img_folder, f'{img_idx}.{self.ext}')
        img = Image.open(img_path).convert('RGB')

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class CelebD2CAttrFewshotDataset(CelebAttrFewshotDataset):
    """
    Similar to CelebAttrFewshotDataset but with D2C-specific cropping.
    """
    def __init__(self,
                 cls_name: str,
                 K: int,
                 img_folder: str,
                 img_size: int = 64,
                 ext: str = 'jpg',
                 seed: int = 0,
                 only_cls_name: Optional[str] = None,
                 only_cls_value: Optional[int] = None,
                 all_neg: bool = False,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__(
            cls_name=cls_name,
            K=K,
            img_folder=img_folder,
            img_size=img_size,
            ext=ext,
            seed=seed,
            only_cls_name=only_cls_name,
            only_cls_value=only_cls_value,
            all_neg=all_neg,
            do_augment=do_augment,
            do_transform=do_transform,
            do_normalize=do_normalize,
            d2c=True
        )


class CelebHQAttrDataset(Dataset):

    id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path: str = os.path.expanduser('datasets/celebahq256.lmdb'),
                 image_size: Optional[int] = None,
                 attr_path: str = os.path.expanduser(
                     'datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
                 original_resolution: int = 256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        self.transform = make_transform(
            image_size=image_size if image_size else original_resolution,
            do_augment=do_augment,
            do_transform=do_transform,
            do_normalize=do_normalize
        )

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)

    def pos_count(self, cls_name: str) -> int:
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name: str) -> int:
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]
        img = img['image']  # Assuming 'image' key exists

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebHQAttrFewshotDataset(Dataset):
    def __init__(self,
                 cls_name: str,
                 K: int,
                 path: str,
                 image_size: int,
                 original_resolution: int = 256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.cls_name = cls_name
        self.K = K
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        self.transform = make_transform(
            image_size=image_size,
            do_augment=do_augment,
            do_transform=do_transform,
            do_normalize=do_normalize
        )

        self.df = pd.read_csv(f'data/celebahq_fewshots/K{K}_{cls_name}.csv', index_col=0)

    def pos_count(self, cls_name: str) -> int:
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name: str) -> int:
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]
        img = img['image']  # Assuming 'image' key exists

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class Repeat(Dataset):
    def __init__(self, dataset: Dataset, new_len: int) -> None:
        """
        Repeat a dataset to a new length.

        Args:
            dataset (Dataset): The original dataset.
            new_len (int): The desired length after repetition.
        """
        super().__init__()
        self.dataset = dataset
        self.original_len = len(dataset)
        self.new_len = new_len

    def __len__(self):
        return self.new_len

    def __getitem__(self, index: int) -> Dict:
        index = index % self.original_len
        return self.dataset[index]


# ---------------------------
# New Dataset Classes for EEG and Conditional DDIM
# ---------------------------

class EEGDataset(Dataset):
    """
    Dataset class for EEG data (Semantic Encoder Training).
    Assumes EEG data is stored in LMDB with each entry containing:
        - 'eeg': Tensor representing EEG signals
    """
    def __init__(self,
                 path: str,
                 transform: Optional[Callable] = None):
        """
        Initialize the EEGDataset.

        Args:
            path (str): Path to the EEG LMDB dataset.
            transform (Optional[Callable], optional): Transformation to apply. Defaults to None.
        """
        self.data = BaseLMDB(path, original_resolution=1, zfill=5)  # Assuming resolution=1 for EEG
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        store_data = self.data[index]
        eeg = store_data['eeg']  # Assuming 'eeg' key exists and is a list or numpy array
        eeg = torch.tensor(eeg, dtype=torch.float32)
        if self.transform:
            eeg = self.transform(eeg)
        return eeg


class ConditionalEEGImageDataset(Dataset):
    """
    Dataset class for Conditional DDIM Training (EEG + Image Data).
    Assumes paired data is stored in LMDB with each entry containing:
        - 'eeg': Tensor representing EEG signals
        - 'image': PIL Image
    """
    def __init__(self,
                 path: str,
                 image_size: int = 64,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 crop_d2c: bool = False):
        """
        Initialize the ConditionalEEGImageDataset.

        Args:
            path (str): Path to the paired EEG and Image LMDB dataset.
            image_size (int, optional): Desired image size after transformations. Defaults to 64.
            do_augment (bool, optional): Whether to apply data augmentation. Defaults to True.
            do_transform (bool, optional): Whether to convert images to tensors. Defaults to True.
            do_normalize (bool, optional): Whether to normalize images. Defaults to True.
            crop_d2c (bool, optional): Whether to apply D2C-specific cropping. Defaults to False.
        """
        self.data = BaseLMDB(path, original_resolution=256, zfill=5)  # Adjust zfill based on LMDB key format
        self.image_size = image_size
        self.transform = make_transform(
            image_size=image_size,
            do_augment=do_augment,
            do_transform=do_transform,
            do_normalize=do_normalize,
            crop_d2c=crop_d2c
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        store_data = self.data[index]
        eeg = store_data['eeg']  # Assuming 'eeg' key exists and is a list or numpy array
        image = store_data['image']  # Assuming 'image' key exists and is a PIL Image

        eeg = torch.tensor(eeg, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)

        return eeg, image
