from typing import Literal

import torch

from data_manager.code.load_dataset import load_dataset
from pl_modules import VisionTransformerPL, DataModulePL
from vision_transformer import VisionTransformer

import pytorch_lightning as pl


def model_validate(split: Literal["train", "test"] = "train"):
    dataset = load_dataset(split=split, verbose=True)

    checkpoint_path = "lightning_logs/version_19/checkpoints/epoch=82-step=10457.ckpt"

    model = VisionTransformer(
        image_size=300,
        patch_size=30,
        num_classes=3,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        apply_rotary_emb=True,
        pool="mean",
    )

    pl_module = VisionTransformerPL.load_from_checkpoint(checkpoint_path, model=model)

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
    model_validate(split="train")
