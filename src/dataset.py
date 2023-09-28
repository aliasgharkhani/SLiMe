import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from src.utils import get_random_crop_coordinates
from glob import glob
import os


class Dataset(TorchDataset):
    def __init__(
        self,
        data_dir,
        train=True,
        mask_size=512,
        num_parts=1,
        min_crop_ratio=0.5,
        dataset_name: str = "sample",
    ):
        self.image_paths = sorted(glob(os.path.join(data_dir, "*.png")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        self.train = train
        self.mask_size = mask_size
        self.num_parts = num_parts
        self.min_crop_ratio = min_crop_ratio
        self.train_transform_1 = A.Compose(
            [
                A.Resize(512, 512),
                A.HorizontalFlip(),
                A.GaussianBlur(blur_limit=(1, 5)),
            ]
        )
        if dataset_name == "celeba":
            rotation_range = (-10, 10)
        else:
            rotation_range = (-30, 30)
        self.train_transform_2 = A.Compose(
            [
                A.Resize(512, 512),
                A.Rotate(
                    rotation_range,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                ToTensorV2(),
            ]
        )
        self.current_part_idx = 0
        self.test_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if len(image.shape) > 2 and image.shape[2] == 4:
            # convert the image from RGBA2RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if self.train:
            mask = np.load(self.mask_paths[idx])
            result = self.train_transform_1(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]
            original_mask_size = np.where(mask == self.current_part_idx, 1, 0).sum()
            mask_is_included = False
            while not mask_is_included:
                x_start, x_end, y_start, y_end = get_random_crop_coordinates(
                    (self.min_crop_ratio, 1), 512, 512
                )
                aux_mask = mask[y_start:y_end, x_start:x_end]
                if (
                    original_mask_size == 0
                    or np.where(aux_mask == self.current_part_idx, 1, 0).sum()
                    / original_mask_size
                    > 0.3
                ):
                    mask_is_included = True
            image = image[y_start:y_end, x_start:x_end]
            result = self.train_transform_2(image=image, mask=aux_mask)
            mask, image = result["mask"], result["image"]
            mask = torch.nn.functional.interpolate(
                mask[None, None, ...].type(torch.float),
                self.mask_size,
                mode="nearest",
            )[0, 0]
            self.current_part_idx += 1
            self.current_part_idx = self.current_part_idx % self.num_parts
            return image / 255, mask
        else:
            if len(self.mask_paths) > 0:
                mask = np.load(self.mask_paths[idx])
                result = self.test_transform(image=image, mask=mask)
                mask = result["mask"]
                mask = torch.nn.functional.interpolate(
                    mask[None, None, ...].type(torch.float),
                    self.mask_size,
                    mode="nearest",
                )[0, 0]
            else:
                result = self.test_transform(image=np.array(image))
                mask = 0
            image = result["image"]
            return image / 255, mask

    def __len__(self):
        return len(self.image_paths)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str = "./data",
        val_data_dir: str = "./data",
        test_data_dir: str = "./data",
        batch_size: int = 1,
        train_mask_size: int = 256,
        test_mask_size: int = 256,
        num_parts: int = 2,
        min_crop_ratio: float = 0.5,
        dataset_name: str = "sample",
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.train_mask_size = train_mask_size
        self.test_mask_size = test_mask_size
        self.num_parts = num_parts
        self.min_crop_ratio = min_crop_ratio
        self.dataset_name = dataset_name

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = Dataset(
                data_dir=self.train_data_dir,
                train=True,
                mask_size=self.train_mask_size,
                num_parts=self.num_parts,
                min_crop_ratio=self.min_crop_ratio,
                dataset_name=self.dataset_name,
            )
            self.val_dataset = Dataset(
                data_dir=self.val_data_dir,
                train=False,
                mask_size=self.test_mask_size,
            )
        elif stage == "test":
            self.test_dataset = Dataset(
                data_dir=self.test_data_dir,
                train=False,
                mask_size=self.test_mask_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False
        )
