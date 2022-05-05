import torch.nn as nn
import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from gi_tract_seg.data.utils import rle_encode
from pytorch_toolbelt.inference import (
    GeneralizedTTA,
    d4_image_augment,
    d4_image_deaugment,
)
from gi_tract_seg import utils

PREDICTION_CLASSES = ["large_bowel", "small_bowel", "stomach"]


class SimpleInferencer(object):
    def __init__(
        self, models: List[nn.Module], threshold: float = 0.5, tta: bool = False
    ):
        self.models = models if not tta else self.make_tta_models(models)
        self.threshold = threshold
        self.tta = tta

    def make_tta_models(self, models):
        return [
            GeneralizedTTA(
                model, augment_fn=d4_image_augment, deaugment_fn=d4_image_deaugment
            )
            for model in models
        ]

    def infer_on_dataset(self, test_dataset: Dataset) -> pd.DataFrame:
        ids = []
        classes = []
        rles = []
        for idx in tqdm(range(len(test_dataset))):
            with torch.no_grad():
                # inference per individual image
                image = test_dataset[idx]["image"][None, ...].cuda()
                output = [model(image) for model in self.models]
                output = torch.stack(output).mean(dim=0)
                output = torch.nn.Sigmoid()(output).cpu().numpy()
            output = (output >= self.threshold).astype("uint8")
            root_shape = (
                test_dataset.data.iloc[idx]["slice_h"],
                test_dataset.data.iloc[idx]["slice_w"],
            )

            encoded_masks = []
            for cls_idx in range(len(PREDICTION_CLASSES)):
                pred_arr = np.round(
                    cv2.resize(
                        output[0, cls_idx, :, :],
                        root_shape,
                        interpolation=cv2.INTER_NEAREST,
                    )
                ).astype("uint8")
                rle_str = rle_encode(pred_arr)
                encoded_masks.append(rle_str)
            ids.extend([test_dataset.df.iloc[idx]["id"]] * 3)
            classes.extend(PREDICTION_CLASSES)
            rles.extend(encoded_masks)

        df = pd.DataFrame()
        df["id"] = ids
        df["class"] = classes
        df["predicted"] = rles
        return df
