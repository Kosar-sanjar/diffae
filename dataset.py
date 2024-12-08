import os
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUNClass
import torch
import pandas as pd

import torchvision.transforms.functional as Ftrans
import numpy as np
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import pickle

class EEGFFHQlmdbDataset(Dataset):
    def __init__(
        self,
        path,
        image_size,
        transform: bool = True,
        normalize: bool = True,
    ):
        """
        Dataset class to load EEG signals and corresponding images from an LMDB database.
        
        Args:
            path (str): Path to the LMDB database.
            image_size (int): Desired image size after resizing.
            transform (bool): Whether to apply image transformations.
            normalize (bool): Whether to normalize the images.
        """
        super().__init__()
        self.path = path
        self.image_size = image_size
        self.transform_flag = transform
        self.normalize_flag = normalize

        # Initialize the BaseLMDB
        self.data = BaseLMDB(path, original_resolution=image_size)
        self.length = len(self.data)

        # Define image transformations
        transform_list = []
        if self.transform_flag:
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
            ])
        transform_list.append(transforms.ToTensor())
        if self.normalize_flag:
            transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), 
                                                        (0.229, 0.224, 0.225)))  # ImageNet statistics
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Retrieve the serialized data from LMDB
        serialized_data = self.data[index]
        storedata = pickle.loads(serialized_data)
        
        # Extract EEG data and image
        eeg = storedata["data"]        # Shape: (channels, time_steps)
        image = storedata["image"]    # PIL Image

        # Convert EEG data to tensor
        eeg_tensor = torch.tensor(eeg).float()  # Shape: (channels, time_steps)
        
        # Apply image transformations
        if self.transform is not None:
            image = self.transform(image)  # Shape: (3, H, W)
        
        return {
            "eeg": eeg_tensor,    # EEG signal
            "image": image,       # Corresponding image
            "label": storedata.get("label", -1),  # Label (if applicable)
            "subject": storedata.get("subject", -1)  # Subject info
        }

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg'],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=False,
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

        transform = [
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
        ]
        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('1')
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}

class EEGImageDataset(Dataset):
    def __init__(
        self,
        paths,
        image_size,
        do_transform: bool = True,

    ):
        super().__init__()
        self.paths = paths
        self.image_size = image_size

        # relative paths (make it shorter, saves memory and faster to sort)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('1')
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}



class NumpyDataset(Dataset):
    def __init__(
        self,
        arrays,
        do_transform: bool = True,
    ):
        super().__init__()
        self.arrays = arrays
        transform = []
        if do_transform:
            transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, index):
        img = self.arrays[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'EEG': img, 'index': index}


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5):
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
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            # key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode(
            #     'utf-8')
            key = f'data-{str(index).zfill(self.zfill)}'.encode('utf-8')
            
            img_bytes = txn.get(key)

        # 3d (RGB) eeg input
        # buffer = np.frombuffer(img_bytes).reshape((128,128,3))
        # normalized_data = (buffer - np.min(buffer)) / (np.max(buffer) - np.min(buffer))
        # scaled_data = normalized_data * 255
        # integer_data = scaled_data.astype(np.uint8)
        # img = Image.fromarray(integer_data,"RGB")
        
        # 2d (RGB) eeg input 128x128
        # buffer = np.frombuffer(img_bytes).reshape((128,128))
        # normalized_data = (buffer - np.min(buffer)) / (np.max(buffer) - np.min(buffer))
        # scaled_data = normalized_data * 255
        # integer_data = scaled_data.astype(np.uint8)
        # img = Image.fromarray(integer_data)

        # 2d (RGB) eeg input 128x496
        # buffer = np.frombuffer(img_bytes).reshape((128,496))
        # normalized_data = (buffer - np.min(buffer)) / (np.max(buffer) - np.min(buffer))
        # scaled_data = normalized_data * 255
        # integer_data = scaled_data.astype(np.uint8)
        # img = Image.fromarray(integer_data)

        # 2d (RGB) eeg input 128x440 not image it
        buffer = np.array(pickle.loads(img_bytes))
        # buffer = np.frombuffer(img_bytes).reshape((128,400))
        
        # normalized_data = (buffer - np.min(buffer)) / (np.max(buffer) - np.min(buffer))
        # normalized_data = buffer - np.min(buffer)
        # scaled_data = normalized_data * 1
        integer_data = buffer.astype(np.float32)
        # img = Image.fromarray(integer_data)
        return integer_data
        
        # img = Image.fromarray(buffer,"RGB")

        # return img

        # img = Image.open(buffer)
        # return img


def make_transform(
    image_size,
    flip_prob=0.5,
    crop_d2c=False,
):
    if crop_d2c:
        transform = [
            d2c_crop(),
            # transforms.Resize(image_size),
        ]
    else:
        transform = [
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
        ]
    # transform.append(transforms.RandomHorizontalFlip(p=flip_prob))
    transform.append(transforms.ToTensor())
    # transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)
    return transform



class FFHQlmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/ffhq256.lmdb'),
                 image_size=256,
                 original_resolution=256,
                 split=None,
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
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == 'test':
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            # transforms.Resize(image_size),
        ]
        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


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


class CelebAlmdb(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 image_size,
                 original_resolution=128,
                 split=None,
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
            raise NotImplementedError()

        if crop_d2c:
            transform = [
                d2c_crop(),
                # transforms.Resize(image_size),
            ]
        else:
            transform = [
                # transforms.Resize(image_size),
                # transforms.CenterCrop(image_size),
            ]

        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Horse_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/horse256.lmdb'),
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
        ]
        # if do_augment:
            # transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Bedroom_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/bedroom256.lmdb'),
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
        ]
        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
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
                 folder,
                 image_size=64,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext='png',
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 d2c: bool = False):
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

        if d2c:
            transform = [
                d2c_crop(),
                # transforms.Resize(image_size),
            ]
        else:
            transform = [
                # transforms.Resize(image_size),
                # transforms.CenterCrop(image_size),
            ]
        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)
            self.df = self.df[self.df.index.isin(paths)]

        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.folder, name)
        img = Image.open(path)

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebD2CAttrDataset(CelebAttrDataset):
    """
    the dataset is used in the D2C paper. 
    it has a specific crop from the original CelebA.
    """
    def __init__(self,
                 folder,
                 image_size=64,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext='jpg',
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 d2c: bool = True):
        super().__init__(folder,
                         image_size,
                         attr_path,
                         ext=ext,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         d2c=d2c)


class CelebAttrFewshotDataset(Dataset):
    def __init__(
        self,
        cls_name,
        K,
        img_folder,
        img_size=64,
        ext='png',
        seed=0,
        only_cls_name: str = None,
        only_cls_value: int = None,
        all_neg: bool = False,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        d2c: bool = False,
    ) -> None:
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
                # transforms.Resize(img_size),
            ]
        else:
            transform = [
                # transforms.Resize(img_size),
                # transforms.CenterCrop(img_size),
            ]
        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.img_folder, name)
        img = Image.open(path)

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class CelebD2CAttrFewshotDataset(CelebAttrFewshotDataset):
    def __init__(self,
                 cls_name,
                 K,
                 img_folder,
                 img_size=64,
                 ext='jpg',
                 seed=0,
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 all_neg: bool = False,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 is_negative=False,
                 d2c: bool = True) -> None:
        super().__init__(cls_name,
                         K,
                         img_folder,
                         img_size,
                         ext=ext,
                         seed=seed,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         all_neg=all_neg,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         d2c=d2c)
        self.is_negative = is_negative


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
                 path=os.path.expanduser('datasets/celebahq256.lmdb'),
                 image_size=None,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        transform = [
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
        ]
        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebHQAttrFewshotDataset(Dataset):
    def __init__(self,
                 cls_name,
                 K,
                 path,
                 image_size,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.cls_name = cls_name
        self.K = K
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        transform = [
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
        ]
        # if do_augment:
        #     transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        # if do_normalize:
        #     transform.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        self.df = pd.read_csv(f'data/celebahq_fewshots/K{K}_{cls_name}.csv',
                              index_col=0)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class Repeat(Dataset):
    def __init__(self, dataset, new_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.original_len = len(dataset)
        self.new_len = new_len

    def __len__(self):
        return self.new_len

    def __getitem__(self, index):
        index = index % self.original_len
        return self.dataset[index]
