from typing import Any, Dict, List, Optional

import torch
import torch.optim.lr_scheduler as scheduler
import torch_optimizer as optim
import torchmetrics
from pytorch_lightning import LightningModule

from gi_tract_seg.models.metrics import DiceMeter


class GITractSegmentatonLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer_name: str,
        lr_scheduler_name: str,
        optimizer_params: Optional[Dict] = None,
        lr_scheduler_params: Optional[Dict] = None,
        lr_scheduler_monitor: Optional[str] = None,
        aux_loss_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.criterion = criterion
        self.aux_loss_multiplier = aux_loss_multiplier
        self.aux_criterion = torch.nn.BCEWithLogitsLoss()
        # we will only track dice during training

        self.train_dice = DiceMeter(
            mode="multilabel", class_names=["lb", "sb", "st"], prefix="train/dice"
        )
        self.val_dice = DiceMeter(
            mode="multilabel", class_names=["lb", "sb", "st"], prefix="val/dice"
        )
        self.test_dice = DiceMeter(
            mode="multilabel", class_names=["lb", "sb", "st"], prefix="test/dice"
        )
        self.train_roc_auc = torchmetrics.AUROC(
            num_classes=3, average="macro", compute_on_step=False
        )
        self.val_roc_auc = torchmetrics.AUROC(
            num_classes=3, average="macro", compute_on_step=False
        )
        self.test_roc_auc = torchmetrics.AUROC(
            num_classes=3, average="macro", compute_on_step=False
        )

    def forward(self, x: torch.Tensor):
        output = self.net(x)
        if type(output) is tuple:
            return {"logits_mask": output[0], "logits_labels": output[1]}
        else:
            return {"logits_mask": output}

    def step(self, batch: Any):
        logits = self.forward(batch["image"])
        loss = self.criterion(logits["logits_mask"], batch["mask"])
        if "logits_labels" in logits.keys() and self.aux_loss_multiplier is not None:
            # need to have aux loss
            aux_loss = self.aux_criterion(logits["logits_labels"], batch["labels"])
            loss = loss + aux_loss * self.aux_loss_multiplier
        return loss, logits

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits = self.step(batch)
        self.train_dice.update(logits["logits_mask"], batch["mask"])
        if self.aux_loss_multiplier is not None:
            self.train_roc_auc.update(logits["logits_labels"], batch["labels"].int())
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        train_dice = self.train_dice.compute()
        self.train_dice.reset()
        self.log_dict(
            train_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.aux_loss_multiplier is not None:
            train_roc_auc = self.train_roc_auc.compute()
            self.train_roc_auc.reset()
            self.log(
                "train/roc_auc",
                train_roc_auc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits = self.step(batch)
        self.val_dice.update(logits["logits_mask"], batch["mask"])
        if self.aux_loss_multiplier is not None:
            self.val_roc_auc.update(logits["logits_labels"], batch["labels"].int())
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        val_dice = self.val_dice.compute()
        self.val_dice.reset()
        self.log_dict(
            val_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.aux_loss_multiplier is not None:
            val_roc_auc = self.val_roc_auc.compute()
            self.val_roc_auc.reset()
            self.log(
                "val/roc_auc",
                val_roc_auc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits = self.step(batch)
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer_params = (
            {}
            if self.hparams.optimizer_params is None
            else dict(self.hparams.optimizer_params[0])
        )
        lr_scheduler_params = (
            {}
            if self.hparams.lr_scheduler_params is None
            else dict(self.hparams.lr_scheduler_params[0])
        )

        optimizer = getattr(optim, self.hparams.optimizer_name)(
            params=self.parameters(), **optimizer_params
        )
        lr_scheduler = getattr(scheduler, self.hparams.lr_scheduler_name)(
            optimizer=optimizer, **lr_scheduler_params
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self.hparams.lr_scheduler_monitor,
            },
        }
