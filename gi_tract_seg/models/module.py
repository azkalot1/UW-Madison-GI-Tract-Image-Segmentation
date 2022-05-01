from typing import Any, List, Dict, Optional
import torch
from pytorch_lightning import LightningModule
import torch_optimizer as optim
import torch.optim.lr_scheduler as scheduler


class GITractSegmentatonLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_name: str,
        lr_scheduler_name: str,
        optimizer_params: Optional[Dict] = None,
        lr_scheduler_params: Optional[Dict] = None,
        lr_scheduler_monitor: Optional[str] = None
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        logits = self.forward(batch['image'])
        loss = self.criterion(logits, batch['mask'])
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
            )
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
            )
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer_params = {} if self.hparams.optimizer_params is None\
            else self.hparams.optimizer_params
        lr_scheduler_params = {} if self.hparams.lr_scheduler_params is None\
            else self.hparams.lr_scheduler_params

        optimizer = getattr(optim, self.hparams.optimizer_name)(
            params=self.parameters(),
            **optimizer_params
            )
        lr_scheduler = getattr(scheduler, self.hparams.lr_scheduler_name)(
            optimizer=optimizer,
            **lr_scheduler_params
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler":
                {
                    "scheduler": lr_scheduler,
                    "monitor": self.hparams.lr_scheduler_monitor
                }
        }
