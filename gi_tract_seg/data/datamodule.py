from typing import Optional
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from gi_tract_seg.data.dataset import GITractDataset
from gi_tract_seg import utils
import albumentations as A
from torch.utils.data import Sampler
from typing import Sequence, Iterator
import torch

log = utils.get_logger(__name__)


class SelectiveWeightenedBatchSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        weights: Sequence,
        sampling_non_empty: float,
        replacement: bool = True,
        drop_last: bool = False,
        generator=None,
    ) -> None:
        self.weights_non_empty = torch.as_tensor(weights, dtype=torch.double)
        self.weights_random = torch.as_tensor(np.ones_like(weights), dtype=torch.double)
        self.sampling_non_empty = torch.tensor(sampling_non_empty)

        self.batch_size = batch_size
        self.replacement = replacement
        self.generator = generator
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[int]:
        for idx in range(len(self)):
            if torch.rand(1)[0] <= self.sampling_non_empty:
                rand_tensor = torch.multinomial(
                    self.weights_non_empty,
                    self.batch_size,
                    self.replacement,
                    generator=self.generator,
                )
            else:
                rand_tensor = torch.multinomial(
                    self.weights_random,
                    self.batch_size,
                    self.replacement,
                    generator=self.generator,
                )
            yield rand_tensor.tolist()

    def __len__(self) -> int:
        return (
            self.weights_non_empty.size()[0] + self.batch_size - 1
        ) // self.batch_size


class GITractDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        images_path: str,
        masks_path: str,
        val_fold: int,
        test_fold: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_augmentations=A.Compose([A.Resize(256, 256)]),
        val_augmentations=A.Compose([A.Resize(256, 256)]),
        shuffle_train: bool = True,
        keep_non_empty: bool = False,
        apply_filters: bool = False,
        use_pseudo_3d: bool = False,
        channels: Optional[int] = None,
        stride: Optional[int] = None,
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
        self.shuffle_train = shuffle_train
        self.keep_non_empty = keep_non_empty
        self.apply_filters = apply_filters
        self.use_pseudo_3d = use_pseudo_3d
        self.channels = channels
        self.stride = stride

    @property
    def train_augmentations(self):
        return self._train_augmentations

    @property
    def val_augmentations(self):
        return self._val_augmentations

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        log.info("Setting up data in datamodule")
        non_train_folds = [self.val_fold]
        self.val_data = self.data.loc[self.data["fold"] == self.val_fold, :]

        if self.test_fold is not None:
            self.test_data = self.data.loc[self.data["fold"] == self.test_fold, :]
            non_train_folds += [self.test_fold]

        self.train_data = self.data.loc[~self.data["fold"].isin(non_train_folds), :]

        log.info(f"Non train folds: {non_train_folds}")
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.train_dataset = GITractDataset(
                self.train_data,
                self.images_path,
                self.masks_path,
                self.train_augmentations,
                self.apply_filters,
                self.use_pseudo_3d,
                self.channels,
                self.stride,
                self.keep_non_empty,
            )
            self.val_dataset = GITractDataset(
                self.val_data,
                self.images_path,
                self.masks_path,
                self.val_augmentations,
                self.apply_filters,
                self.use_pseudo_3d,
                self.channels,
                self.stride,
                self.keep_non_empty,
            )
            if self.test_fold is not None:
                self.test_dataset = GITractDataset(
                    self.test_data,
                    self.images_path,
                    self.masks_path,
                    self.val_augmentations,
                    self.apply_filters,
                    self.use_pseudo_3d,
                    self.channels,
                    self.stride,
                    self.keep_non_empty,
                )

    def train_dataloader(self):
        batch_sampler = SelectiveWeightenedBatchSampler(
            batch_size=self.batch_size,
            weights=1 - self.train_dataset.data["empty"].values,
            sampling_non_empty=0.8,
        )
        return DataLoader(
            dataset=self.train_dataset,
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # shuffle=self.shuffle_train,
            batch_sampler=batch_sampler,
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
