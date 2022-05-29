from typing import Literal

import torch
from torchvision.models import vgg19

from data_manager.code.load_dataset import load_dataset
from pl_modules import HandClassifierPL, DataModulePL

import pytorch_lightning as pl


def model_validate(split: Literal["train", "test"] = "train"):
    dataset = load_dataset(split=split, verbose=True)

    checkpoint_path = "lightning_logs/version_27/checkpoints/epoch=0-step=125.ckpt"

    model = vgg19(pretrained=False, num_classes=3)

    pl_module = HandClassifierPL.load_from_checkpoint(checkpoint_path, model=model)

    gpus = -1 if torch.cuda.device_count() else 0

    trainer = pl.Trainer(
        precision=32,
        reload_dataloaders_every_n_epochs=1,
        gpus=gpus
    )

    datamodule = DataModulePL(
        dataset,
        batch_size=16,
        num_workers=4
    )

    trainer.validate(pl_module, datamodule)


if __name__ == '__main__':
    model_validate(split="test")
