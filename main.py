import torch
import pytorch_lightning as pl

from data_manager import load_dataset
from pl_modules import VisionTransformerPL, DataModulePL
from vision_transformer import VisionTransformer


def main():
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
        dropout=0.1,
        emb_dropout=0.1,
        apply_rotary_emb=True,
    )

    num_trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num of trainable params = {num_trainable_param}")

    pl_module = VisionTransformerPL(model)

    datamodule = DataModulePL(
        dataset,
        batch_size=32,
        num_workers=0
    )

    trainer.fit(pl_module, datamodule)


if __name__ == '__main__':
    main()

