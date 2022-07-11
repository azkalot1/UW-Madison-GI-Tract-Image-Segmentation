from functools import partial
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch
from torchmetrics import Metric

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def to_numpy(x: Union[torch.Tensor, np.ndarray, Any, None]) -> Union[np.ndarray, None]:
    """
    Convert whatever to numpy array. None value returned as is.
    Args:
        :param x: List, tuple, PyTorch tensor or numpy array
    Returns:
        :return: Numpy array
    """
    if x is None:
        return None
    elif torch.is_tensor(x):
        return x.data.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (Iterable, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")


def binary_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold: Optional[float] = None,
    nan_score_on_empty=False,
    eps: float = 1e-7,
    ignore_index=None,
) -> float:
    # Source: pytorch_toolbelt
    assert mode in {"dice", "iou"}

    # Make binary predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    if ignore_index is not None:
        mask = (y_true != ignore_index).to(y_true.dtype)
        y_true = y_true * mask
        y_pred = y_pred * mask

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if mode == "dice":
        score = (2.0 * intersection) / (cardinality + eps)
    else:
        score = intersection / (cardinality - intersection + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score


def multiclass_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
    ignore_index=None,
):
    # Source: pytorch_toolbelt
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        y_pred_i = (y_pred == class_index).float()
        y_true_i = (y_true == class_index).float()
        if ignore_index is not None:
            not_ignore_mask = (y_true != ignore_index).float()
            y_pred_i *= not_ignore_mask
            y_true_i *= not_ignore_mask

        iou = binary_dice_iou_score(
            y_pred=y_pred_i,
            y_true=y_true_i,
            mode=mode,
            nan_score_on_empty=nan_score_on_empty,
            threshold=threshold,
            eps=eps,
        )
        ious.append(iou)

    return ious


def multilabel_dice_iou_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
    ignore_index=None,
):
    # Source: pytorch_toolbelt
    ious = []
    num_classes = y_pred.shape[0]

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_dice_iou_score(
            y_pred=y_pred[class_index],
            y_true=y_true[class_index],
            mode=mode,
            threshold=threshold,
            nan_score_on_empty=nan_score_on_empty,
            eps=eps,
            ignore_index=ignore_index,
        )
        ious.append(iou)

    return ious


class DiceMeter(Metric):
    def __init__(
        self,
        mode: str,
        metric="dice",
        class_names=None,
        classes_of_interest=None,
        nan_score_on_empty=True,
        prefix: str = None,
        ignore_index=None,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if mode not in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}:
            raise ValueError("Not supported mode")

        if prefix is None:
            prefix = metric

        self.mode = mode
        self.prefix = prefix
        self.class_names = class_names
        self.classes_of_interest = classes_of_interest
        self.scores = []
        if self.mode == BINARY_MODE:
            self.score_fn = partial(
                binary_dice_iou_score,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                mode=metric,
                ignore_index=ignore_index,
            )

        if self.mode == MULTICLASS_MODE:
            self.score_fn = partial(
                multiclass_dice_iou_score,
                mode=metric,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                classes_of_interest=self.classes_of_interest,
                ignore_index=ignore_index,
            )

        if self.mode == MULTILABEL_MODE:
            self.score_fn = partial(
                multilabel_dice_iou_score,
                mode=metric,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                classes_of_interest=self.classes_of_interest,
                ignore_index=ignore_index,
            )

    def reset(self):
        self.scores = []

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_size = target.size(0)
        score_per_image = []
        for image_index in range(batch_size):
            score_per_class = self.score_fn(
                y_pred=preds[image_index], y_true=target[image_index]
            )
            score_per_class = to_numpy(score_per_class).reshape(-1)
            score_per_image.append(score_per_class)
        self.scores.extend(score_per_image)

    def compute(self):
        computed_scores = {}
        scores = np.array(self.scores)
        mean_per_class = np.nanmean(scores, axis=1)
        mean_score = np.nanmean(mean_per_class, axis=0)
        computed_scores[self.prefix] = mean_score
        if self.mode in {MULTICLASS_MODE, MULTILABEL_MODE}:
            num_classes = scores.shape[1]
            class_names = self.class_names
            if class_names is None:
                class_names = [f"class_{i}" for i in range(num_classes)]

            scores_per_class = np.nanmean(scores, axis=0)
            for name, s_p_c in zip(class_names, scores_per_class):
                computed_scores[name + "_" + self.prefix] = float(s_p_c)
        return computed_scores
