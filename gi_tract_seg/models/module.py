from typing import Any, List, Dict, Optional
import torch
from pytorch_lightning import LightningModule
import torch_optimizer as optim
import torch.optim.lr_scheduler as scheduler
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
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.criterion = criterion
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

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        logits = self.forward(batch["image"])
        gt_mask = batch["mask"]
        loss = self.criterion(logits, gt_mask)
        return loss, logits, gt_mask

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, gt_mask = self.step(batch)
        self.train_dice.update(logits, gt_mask)
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        train_dice = self.train_dice.compute()
        self.train_dice.reset()
        self.log_dict(
            train_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, gt_mask = self.step(batch)
        self.val_dice.update(logits, gt_mask)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        val_dice = self.val_dice.compute()
        self.val_dice.reset()
        self.log_dict(
            val_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits, gt_mask = self.step(batch)
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
