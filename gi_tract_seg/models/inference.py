import math
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_toolbelt.inference.tta import (GeneralizedTTA, d4_image_augment,
                                            d4_image_deaugment)
from torch.utils.data import Dataset
from tqdm import tqdm

from gi_tract_seg import utils
from gi_tract_seg.data.utils import rle_encode
from gi_tract_seg.models.metrics.dice_meter import (DiceMeter,
                                                    multilabel_dice_iou_score)

PREDICTION_CLASSES = ["large_bowel", "small_bowel", "stomach"]


class SimpleInferencer(object):
    def __init__(
        self, models: List[nn.Module], threshold: float = 0.5, tta: bool = False
    ):
        self.models = models if not tta else self.make_tta_models(models)
        self.threshold = threshold
        self.tta = tta
        self.test_dice = DiceMeter(
            mode="multilabel", class_names=["lb", "sb", "st"], prefix="test/dice"
        )

    def make_tta_models(self, models):
        return [
            GeneralizedTTA(
                model, augment_fn=d4_image_augment, deaugment_fn=d4_image_deaugment
            )
            for model in models
        ]

    def infer_on_dataset(self, test_dataset: Dataset):
        ids = []
        classes = []
        rles = []
        dice_class_1_scores = []
        dice_class_2_scores = []
        dice_class_3_scores = []
        for idx in tqdm(range(len(test_dataset))):
            with torch.no_grad():
                # inference per individual image
                image = test_dataset[idx]["image"][None, ...].cuda()
                mask = test_dataset[idx]["mask"][None, ...].cuda()
                output = [model(image) for model in self.models]
                output = torch.stack(output).mean(dim=0)
                output = torch.nn.Sigmoid()(output).cpu().numpy()
            output = (output >= self.threshold).astype("uint8")

            output_torch = torch.from_numpy(output).cuda()  # Needed to calculate dice

            root_shape = (
                test_dataset.data.iloc[idx]["slice_h"],
                test_dataset.data.iloc[idx]["slice_w"],
            )

            iou = multilabel_dice_iou_score(
                y_true=mask[0],
                y_pred=output_torch[0],
                nan_score_on_empty=True,
                classes_of_interest=None,
            )

            iou_1 = iou[0]
            iou_2 = iou[1]
            iou_3 = iou[2]

            if not math.isnan(iou_1):
                dice_class_1_scores.append(iou_1)
            if not math.isnan(iou_2):
                dice_class_2_scores.append(iou_2)
            if not math.isnan(iou_3):
                dice_class_3_scores.append(iou_3)

            # self.test_dice.update(output["logits_mask"], mask)

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
            ids.extend([test_dataset.data.iloc[idx]["id"]] * 3)
            classes.extend(PREDICTION_CLASSES)
            rles.extend(encoded_masks)

        # Compute dice for each class
        dice_class_1_mean = sum(dice_class_1_scores) / len(dice_class_1_scores)
        dice_class_2_mean = sum(dice_class_2_scores) / len(dice_class_2_scores)
        dice_class_3_mean = sum(dice_class_3_scores) / len(dice_class_3_scores)

        final_dice = (dice_class_1_mean + dice_class_2_mean + dice_class_3_mean) / 3.0

        print(final_dice)

        # dice_score = self.test_dice.compute()
        df = pd.DataFrame()
        df["id"] = ids
        df["class"] = classes
        df["predicted"] = rles
        return df, final_dice
