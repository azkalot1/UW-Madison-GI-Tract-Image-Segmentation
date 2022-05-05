import torch
from torch.utils.data import Dataset
from typing import Optional
import numpy as np
import pandas as pd
import cv2
from os.path import join as path_join
import albumentations as A


class GITractDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        images_path: str,
        masks_path: str,
        transforms: Optional = None,
    ):
        self.data = data
        self.transforms = transforms
        self.images_path = images_path
        self.masks_path = masks_path

    def __len__(self):
        return self.data.shape[0]

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype("float32")  # original is uint16
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)  # scale image to [0, 1]
        img = np.expand_dims(img, -1)
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
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
        }

        return data


class GITractDatasetTest(Dataset):
    def __init__(self, data, transform=A.Compose([A.Resize(256, 256)])):
        self.data = data
        self.image_path = data["f_path"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype("float32")  # original is uint16
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)  # scale image to [0, 1]
        img = np.expand_dims(img, -1)
        return img

    def __getitem__(self, idx):
        image = self._load_image(self.image_path[idx])
        augmented = self.transform(image=image)
        image = augmented["image"]
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        return {"image": torch.tensor(image, dtype=torch.float)}
