import torch

from vision_transformer import VisionTransformer


def test_vision_transformer():
    model = VisionTransformer(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        apply_rotary_emb=True,
    )

    img = torch.randn(1, 3, 256, 256)
    preds = model(img)  # (1, 1000)