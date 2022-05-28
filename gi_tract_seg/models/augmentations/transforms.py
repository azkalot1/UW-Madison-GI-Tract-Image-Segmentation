import albumentations as A
import cv2


def default_valid_transform(resize_size: int = 256):
    return A.Compose(
        [A.Resize(resize_size, resize_size, interpolation=cv2.INTER_NEAREST)], p=1.0
    )


def light_training_transforms(resize_size: int = 256):
    return A.Compose(
        [
            A.Resize(resize_size, resize_size, interpolation=cv2.INTER_NEAREST),
            A.OneOf(
                [
                    A.Transpose(),
                    A.VerticalFlip(),
                    A.HorizontalFlip(),
                    A.RandomRotate90(),
                    A.NoOp(),
                ],
                p=1.0,
            ),
        ]
    )


def medium_training_transforms(resize_size: int = 256):
    return A.Compose(
        [
            A.Resize(resize_size, resize_size, interpolation=cv2.INTER_NEAREST),
            A.OneOf(
                [
                    A.Transpose(),
                    A.VerticalFlip(),
                    A.HorizontalFlip(),
                    A.RandomRotate90(),
                    A.NoOp(),
                ],
                p=1.0,
            ),
            A.CoarseDropout(max_holes=16, max_height=16, max_width=16, p=0.5),
        ]
    )


def heavy_training_transforms(resize_size: int = 256):
    return A.Compose(
        [
            A.RandomScale(scale_limit=0.25, p=0.5),
            A.Resize(resize_size, resize_size, interpolation=cv2.INTER_NEAREST),
            A.OneOf(
                [
                    A.Transpose(),
                    A.VerticalFlip(),
                    A.HorizontalFlip(),
                    A.RandomRotate90(),
                    A.NoOp(),
                ],
                p=1.0,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(),
                    A.GridDistortion(),
                    A.OpticalDistortion(),
                    A.NoOp(),
                    A.ShiftScaleRotate(),
                ],
                p=1.0,
            ),
            A.CoarseDropout(max_holes=16, max_height=16, max_width=16, p=0.5),
        ]
    )


def kaggle_training_transforms(resize_size: int = 256):
    return A.Compose(
        [
            A.Resize(resize_size, resize_size, interpolation=cv2.INTER_NEAREST),
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
                max_height=resize_size // 20,
                max_width=resize_size // 20,
                min_holes=5,
                fill_value=0,
                mask_fill_value=0,
                p=0.5,
            ),
        ]
    )
