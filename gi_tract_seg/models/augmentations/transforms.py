import albumentations as A
import cv2
from typing import List


def default_valid_transform(size: List[int] = [320, 384], padding=False):
    return A.Compose(
        [
            A.PadIfNeeded(
                *size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                position="top_left"
            )
            if padding
            else A.Resize(*size, interpolation=cv2.INTER_NEAREST)
        ]
    )


def light_training_transforms(size: List[int] = [320, 384], padding=False):
    return A.Compose(
        [
            A.PadIfNeeded(
                *size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                position="top_left"
            )
            if padding
            else A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
        ]
    )


def medium_training_transforms(size: List[int] = [320, 384], padding=False):
    return A.Compose(
        [
            A.PadIfNeeded(
                *size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                position="top_left"
            )
            if padding
            else A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes=16, max_height=16, max_width=16, p=0.5),
        ]
    )


def heavy_training_transforms(size: List[int] = [320, 384], padding=False):
    return A.Compose(
        [
            A.OneOf([A.HorizontalFlip(), A.VerticalFlip(), A.Rotate()], p=0.75),
            A.OneOf(
                [
                    A.RandomResizedCrop(*size, p=0.75),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5
                    ),
                ],
                p=0.75,
            ),
            A.PadIfNeeded(
                *size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                position="top_left"
            )
            if padding
            else A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.OneOf(
                [
                    A.ElasticTransform(),
                    A.GridDistortion(),
                    A.OpticalDistortion(),
                ],
                p=0.75,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=size[0] // 20,
                max_width=size[1] // 20,
                min_holes=5,
                fill_value=0,
                mask_fill_value=0,
                p=0.5,
            ),
        ]
    )


def kaggle_training_transforms(size: List[int] = [320, 384], padding=False):
    return A.Compose(
        [
            A.PadIfNeeded(
                *size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                position="top_left"
            )
            if padding
            else A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                ],
                p=0.25,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=size[0] // 20,
                max_width=size[1] // 20,
                min_holes=5,
                fill_value=0,
                mask_fill_value=0,
                p=0.5,
            ),
        ]
    )
