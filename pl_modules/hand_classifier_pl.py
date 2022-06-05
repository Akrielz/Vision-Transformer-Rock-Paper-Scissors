from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn


class HandClassifierPL(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            lr: float = 1e-3,
            image_augmenter: Optional[nn.Module] = None
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.image_augmenter = image_augmenter

        self.accuracy = torchmetrics.Accuracy()

    def __step__(self, imgs, targets, *, mode):
        if self.image_augmenter:
            imgs = torch.cat([imgs, self.image_augmenter(imgs)], dim=0)
            targets = torch.cat([targets, targets], dim=0)

        predicted = self.model(imgs)
        loss = F.cross_entropy(predicted, targets)

        self.accuracy(predicted, targets)

        self.log_dict({f'{mode}_loss': loss})
        self.log_dict({f'{mode}_acc': self.accuracy})

        return loss

    def validation_step(self, batch, batch_idx, *args) -> Optional[STEP_OUTPUT]:
        loss = self.__step__(*batch, mode="val")
        return loss

    def training_step(self, batch, batch_idx, *args) -> STEP_OUTPUT:
        loss = self.__step__(*batch, mode="train")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(lr=self.lr, params=self.model.parameters())
