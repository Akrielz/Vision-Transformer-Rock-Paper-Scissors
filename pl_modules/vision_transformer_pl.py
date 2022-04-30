from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn


class VisionTransformerPL(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model

    def __step__(self, imgs, targets):
        predicted = self.model(imgs)
        loss = F.cross_entropy(predicted, targets)

        return loss

    def validation_step(self, batch, batch_idx, *args) -> Optional[STEP_OUTPUT]:
        loss = self.__step__(*batch)
        self.log_dict({'val_loss': loss})
        return loss

    def training_step(self, batch, batch_idx, *args) -> STEP_OUTPUT:
        loss = self.__step__(*batch)
        self.log_dict({'train_loss': loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(lr=1e-2, params=self.model.parameters())
