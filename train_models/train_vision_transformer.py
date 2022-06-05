import torch
import pytorch_lightning as pl

from data_augmenter.data_augmenter import create_data_augmenter
from data_manager import load_dataset
from pl_modules import HandClassifierPL, DataModulePL
from vision_transformer import VisionTransformer


def model_train():
    dataset = load_dataset(split="train", verbose=True)

    gpus = -1 if torch.cuda.device_count() else 0

    if gpus == 0:
        print("Running on CPU")
    else:
        print("Running on GPU")

    trainer = pl.Trainer(
        precision=32,
        reload_dataloaders_every_n_epochs=1,
        gpus=gpus
    )

    model = VisionTransformer(
        image_size=300,
        patch_size=30,
        num_classes=3,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.3,
        emb_dropout=0.3,
        apply_rotary_emb=True,
        pool="mean",
    )

    augmetor = create_data_augmenter()

    num_trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num of trainable params = {num_trainable_param}")

    pl_module = HandClassifierPL(model=model, lr=2e-5, image_augmenter=augmetor)

    datamodule = DataModulePL(
        dataset,
        batch_size=16,
        num_workers=4
    )

    trainer.fit(pl_module, datamodule)


if __name__ == '__main__':
    model_train()

