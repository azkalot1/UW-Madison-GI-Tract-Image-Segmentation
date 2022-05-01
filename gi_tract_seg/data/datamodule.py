from typing import Optional
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from gi_tract_seg.data.dataset import GITractDataset
from gi_tract_seg import utils
import albumentations as A

log = utils.get_logger(__name__)


class GITractDataModule(LightningDataModule):
    def __init__(self, 
                data_path: str, 
                images_path: str,
                masks_path: str,
                val_fold: int, 
                test_fold: Optional[int] = None,
                batch_size: int = 32,
                num_workers: int = 8,
                pin_memory: bool = True,
                train_augmentations = A.Compose([A.Resize(256, 256)]),
                val_augmentations = A.Compose([A.Resize(256, 256)]),
                ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.data = pd.read_parquet(data_path)
        self.val_fold = val_fold
        self.test_fold = test_fold

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.images_path = images_path
        self.masks_path = masks_path
        self._train_augmentations = train_augmentations
        self._val_augmentations = val_augmentations

    @property
    def train_augmentations(self):
        return self._train_augmentations

    @property
    def val_augmentations(self):
        return self._val_augmentations

    def prepare_data(self):
        non_train_folds = [self.val_fold]
        self.val_data =\
            self.data.loc[self.data['fold'] == self.val_fold, :]

        if self.test_fold is not None:
            self.test_data =\
                self.data.loc[self.data['fold'] == self.test_fold, :]
            non_train_folds += [self.test_fold]

        self.train_data =\
            self.data.loc[~self.data['fold'].isin(non_train_folds), :]

    def setup(self, stage: Optional[str] = None):
        log.info("Setting up data in datamodule")
        if not self.train_dataset\
                and not self.val_dataset\
                and not self.test_dataset:
            self.train_dataset = GITractDataset(
                self.train_data,
                self.images_path,
                self.masks_path,
                self.train_augmentations,
                )
            self.val_dataset = GITractDataset(
                self.val_data,
                self.images_path,
                self.masks_path,
                self.val_augmentations
                )
            if self.test_fold is not None:
                self.test_dataset = GITractDataset(
                    self.test_data,
                    self.images_path,
                    self.masks_path,
                    self.val_augmentations
                    )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
