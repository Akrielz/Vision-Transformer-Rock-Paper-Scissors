import torch

from vision_transformer.vision_transformer import VisionTransformer


def main():
    model = VisionTransformer(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 3, 256, 256)
    preds = model(img)  # (1, 1000)


if __name__ == '__main__':
    main()
