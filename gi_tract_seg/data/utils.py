import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import StratifiedGroupKFold


def get_metadata(row):
    # source: https://www.kaggle.com/code/awsaf49/uwmgi-mask-data
    data = row["id"].split("_")
    case = int(data[0].replace("case", ""))
    day = int(data[1].replace("day", ""))
    slice_ = int(data[-1])
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row


def path2info(row):
    # source: https://www.kaggle.com/code/awsaf49/uwmgi-mask-data
    path = row["image_path"]
    data = path.split("/")
    slice_ = int(data[-1].split("_")[1])
    case = int(data[-3].split("_")[0].replace("case", ""))
    day = int(data[-3].split("_")[1].replace("day", ""))
    width = int(data[-1].split("_")[2])
    height = int(data[-1].split("_")[3])
    row["height"] = height
    row["width"] = width
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row


def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    """
    if not pd.isna(mask_rle):
        s = np.asarray(mask_rle.split(), dtype=int)
        starts = s[0::2] - 1
        lengths = s[1::2]
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)  # Needed to align to RLE direction
    else:
        return np.zeros(shape)


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


def id2mask(df):
    # source: https://www.kaggle.com/code/awsaf49/uwmgi-mask-data
    wh = df[["height", "width"]].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
        cdf = df[df["class"] == class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def compute_segmentation_sizes(df):
    sizes = []
    for idx in range(df.shape[0]):
        segmentation = df.segmentation.values[idx]
        h = df.height.values[idx]
        w = df.width.values[idx]
        sizes.append([rle_decode(seg, (h, w)).sum() for seg in segmentation])
    return sizes


def save_mask(df, image_folder, mask_folder):
    mask = id2mask(df)
    image_path = df.image_path.iloc[0]
    mask_path = image_path.replace(image_folder, mask_folder)
    mask_folder = mask_path.rsplit("/", 1)[0]
    os.makedirs(mask_folder, exist_ok=True)
    cv2.imwrite(mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return mask_path


def merge_ids(df):
    df2 = df.groupby(["id"])["class"].agg(list).to_frame().reset_index()
    df2 = df2.rename(columns={"class": "classes"})
    df2 = df2.merge(df.groupby(["id"])["segmentation"].agg(list), on=["id"])

    df = df.drop(columns=["segmentation", "class"])
    df = df.groupby(["id"]).head(1).reset_index(drop=True)
    df = df.merge(df2, on=["id"])
    df["labels"] = df["segmentation"].apply(lambda x: 1 - pd.isna(x).astype(int))
    df["empty"] = df["labels"].apply(lambda x: (np.array(x) == 0).all().astype(int))

    return df


def create_folds(df, n_splits, random_seed):
    skf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_seed
    )
    df["fold"] = 0
    for fold, (_, val_idx) in enumerate(
        skf.split(X=df, y=df["empty"], groups=df["case"])
    ):
        df.loc[val_idx, "fold"] = fold

    return df
