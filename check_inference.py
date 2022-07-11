import gc
import os
from glob import glob

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch

from gi_tract_seg.data.dataset import GITractDatasetTest
from gi_tract_seg.models.inference import SimpleInferencer
from gi_tract_seg.models.metrics.dice_meter import (DiceMeter,
                                                    multilabel_dice_iou_score)

gc.enable()

PREDICTION_CLASSES = ["large_bowel", "small_bowel", "stomach"]
PATH_TO_MODEL = "./models/last_2d_sampling_heavy_v1_longer_fold1.pt"


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated

    ref: https://www.kaggle.com/stainsby/fast-tested-rle
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def main():
    # Read data
    DEBUG = True
    df = pd.read_csv("./data/raw/sample_submission.csv")
    if df.shape[0] == 0:
        print("Using part of train data as example")
        DEBUG = True
    if DEBUG is True:
        df = pd.read_csv("./data/raw/train.csv")
        df.pop("segmentation")
        df["predicted"] = ""

    # Process data
    ###################################################################################################################
    df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
    df["case_id"] = df["id"].apply(
        lambda x: int(x.split("_", 2)[0].replace("case", ""))
    )

    # 2. Get Day as a column
    df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
    df["day_num"] = df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

    # 3. Get Slice Identifier as a column
    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    if DEBUG:
        TRAIN_DIR = "./data/raw/train"
    else:
        TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

    # Get all training images
    all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
    ""
    p = []
    x = all_train_images[0].rsplit("/", 4)[0]
    for i in range(0, df.shape[0]):
        p.append(
            os.path.join(
                x,
                df["case_id_str"].values[i],
                df["case_id_str"].values[i] + "_" + df["day_num_str"].values[i],
                "scans",
                df["slice_id"].values[i],
            )
        )
    df["_partial_ident"] = p

    p = []
    for i in range(0, len(all_train_images)):
        p.append(str(all_train_images[i].rsplit("_", 4)[0]))

    _tmp_merge_df = pd.DataFrame()
    _tmp_merge_df["_partial_ident"] = p
    _tmp_merge_df["f_path"] = all_train_images

    df = df.merge(_tmp_merge_df, on="_partial_ident").drop(columns=["_partial_ident"])

    # 5. Get slice dimensions from filepath (int in pixels)
    df["slice_h"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["slice_w"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))

    # 6. Pixel spacing from filepath (float in mm)
    df["px_spacing_h"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[4]))

    df1 = df[df.index % 3 == 0]
    df2 = df[df.index % 3 == 1]
    df3 = df[df.index % 3 == 2]
    df = df1.copy()
    # df.pop('class')
    gc.collect()

    del x, df1, df2, df3, _tmp_merge_df
    gc.collect()
    df = df.reset_index(drop=True)

    print(df.head())
    ###################################################################################################################

    data_info = pd.read_parquet("./data/train_df_agg_cleaned.parquet")

    data_info = data_info[data_info.fold == 1]
    print(data_info.head())

    m = df.id.isin(data_info.id)
    df = df[m]

    print(df.shape)
    print(data_info.shape)

    # Load models
    models = []
    model = torch.load(f=PATH_TO_MODEL, map_location="cpu")
    model.eval()
    model.cuda()
    models.append(model)

    # Inference
    test_set = GITractDatasetTest(
        df.reset_index(drop=True),
        use_pseudo_3d=False,
        channels=-1,
        stride=-1,
        transform=A.Compose(
            [
                A.PadIfNeeded(
                    320,
                    384,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    position="top_left",
                )
            ]
        ),
    )
    inferencer = SimpleInferencer(models, 0.5, tta=False)

    pred_df, dice_score = inferencer.infer_on_dataset(test_set)
    print("FINAL DICE SCORE: ", dice_score)
    pred_df.head()

    pred_df.loc[pred_df["predicted"] != "", :].head()
    # Submit
    pred_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
