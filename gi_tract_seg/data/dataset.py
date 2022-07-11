import os
from os.path import join as path_join
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.filters import meijering, sobel
from torch.utils.data import Dataset


class GITractDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        images_path: str,
        masks_path: str,
        transforms: Optional = None,
        apply_filters: bool = False,
        use_pseudo_3d: bool = False,
        channels: Optional[int] = None,
        stride: Optional[int] = None,
        keep_non_empty: bool = True,
    ):
        self.data = data
        self.transforms = transforms
        self.images_path = images_path
        self.masks_path = masks_path
        self.apply_filters = apply_filters
        self.use_pseudo_3d = use_pseudo_3d
        self.channels = channels
        self.stride = stride
        self.keep_non_empty = keep_non_empty
        if self.use_pseudo_3d:
            for i in range(self.channels):
                self.data[f"image_path_{i:02}"] = (
                    self.data.groupby(["case", "day"])["image_path"]
                    .shift(-i * self.stride)
                    .fillna(method="ffill")
                )
            self.data["image_paths"] = self.data[
                [f"image_path_{i:02d}" for i in range(self.channels)]
            ].values.tolist()
            self.data = self.data.drop(
                columns=[f"image_path_{i:02d}" for i in range(self.channels)]
            )

        if self.keep_non_empty:
            self.data = self.data.loc[self.data["empty"] == 0]

    def __len__(self):
        return self.data.shape[0]

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype("float32")  # original is uint16
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)  # scale image to [0, 1]
        if self.apply_filters:
            img_sobel = sobel(img)
            img_meijering = meijering(img)
            img = np.stack([img, img_sobel, img_meijering], -1)
        else:
            img = np.expand_dims(img, -1)
        return img

    def _load_images(self, paths):
        img = [
            cv2.imread(path, cv2.IMREAD_UNCHANGED).astype("float32") for path in paths
        ]
        img = np.stack(img, -1)
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)  # scale image to [0, 1]
        return img

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        mask = mask.astype("float32")
        return mask

    def _get_after_transform(self, image, mask):
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        if self.use_pseudo_3d:
            image_paths = [
                path_join(self.images_path, path)
                for path in self.data.image_paths.values[idx]
            ]
            image = self._load_images(image_paths)
        else:
            image_path = path_join(self.images_path, self.data.image_path.values[idx])
            image = self._load_image(image_path)

        mask_path = path_join(self.masks_path, self.data.mask_path.values[idx])
        mask = self._load_mask(mask_path)

        labels = self.data.labels.values[idx]
        labels = np.array(labels)

        image, mask = self._get_after_transform(image, mask)

        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        mask = mask.transpose(2, 0, 1)
        mask = np.ascontiguousarray(mask)

        data = {
            "image": torch.from_numpy(image).float(),
            "labels": torch.from_numpy(labels).float(),
            "mask": torch.from_numpy(mask),
        }

        return data


class GITractDatasetTest(Dataset):
    def __init__(
        self,
        data,
        transform=A.Compose([A.Resize(256, 256)]),
        apply_filters: bool = False,
        use_pseudo_3d: bool = False,
        channels: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        self.data = data
        self.image_path = data["f_path"]
        self.masks_path = self.image_path
        self.transform = transform
        self.apply_filters = apply_filters
        self.use_pseudo_3d = use_pseudo_3d
        self.channels = channels
        self.stride = stride
        if self.use_pseudo_3d:
            for i in range(self.channels):
                self.data[f"image_path_{i:02}"] = (
                    self.data.groupby(["case_id", "day_num"])["f_path"]
                    .shift(-i * self.stride)
                    .fillna(method="ffill")
                )
            self.data["image_paths"] = self.data[
                [f"image_path_{i:02d}" for i in range(self.channels)]
            ].values.tolist()
            self.data = self.data.drop(
                columns=[f"image_path_{i:02d}" for i in range(self.channels)]
            )

    def __len__(self):
        return len(self.data)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype("float32")  # original is uint16
        max_val = img.max()
        img = img / img.max()  # scale image to [0, 1]
        if self.apply_filters:
            img_sobel = sobel(img)
            img_meijering = meijering(img)
            img = np.stack([img, img_sobel, img_meijering], -1)
        else:
            img = np.expand_dims(img, -1)
        return img

    def _load_images(self, paths):
        img = [self._load_image(path) for path in paths]
        img = np.concatenate(img, -1)
        return img

    @staticmethod
    def _load_mask(path):
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        mask = mask.astype("float32")
        return mask

    def _load_masks(self, paths):
        mask_paths = [self._get_mask_path(path) for path in paths]
        masks = [self._load_mask(mask_path) for mask_path in mask_paths]
        masks = np.concatenate(masks, -1)
        return masks

    @staticmethod
    def _get_mask_path(image_path):
        list_parts = image_path.split("/")
        first_part = "./data/processed/masks"
        last_part = os.path.join(*list_parts[4:])
        mask_path = os.path.join(first_part, last_part)
        return mask_path

    def _get_after_transform(self, image, mask):
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask

    def __getitem__(self, idx):
        if self.use_pseudo_3d:
            image_paths = self.data.image_paths.values[idx]
            image = self._load_images(image_paths)
            mask = self._load_masks(image_paths)
        else:
            image_path = self.data.f_path.values[idx]
            image = self._load_image(image_path)
            mask_path = self._get_mask_path(image_path)
            mask = self._load_mask(mask_path)

        image, mask = self._get_after_transform(image=image, mask=mask)

        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        mask = mask.transpose(2, 0, 1)
        mask = np.ascontiguousarray(mask)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "mask": torch.tensor(mask, dtype=torch.float),
        }
