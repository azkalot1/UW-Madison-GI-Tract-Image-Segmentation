import albumentations as A


def default_valid_transform(resize_size: int = 256):
    return A.Compose([A.Resize(resize_size, resize_size)], p=1.0)


def light_training_transforms(resize_size: int = 256):
    return A.Compose(
        [
            A.Resize(resize_size, resize_size),
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
            A.Resize(resize_size, resize_size),
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
            A.Resize(resize_size, resize_size),
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
