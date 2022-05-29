import torch

from api.convert_image import convert_image
from pl_modules import HandClassifierPL
from vision_transformer import VisionTransformer
from data_manager.code.load_dataset import __LABELS_STR__

if __name__ == "__main__":
    with open("./api/img_base64.txt", "r") as f:
        img_base64 = f.read()

    # Convert from Base 64 to numpy img
    img = convert_image(img_base64)

    # Load model
    checkpoint_path = "lightning_logs/version_24/checkpoints/epoch=4-step=629.ckpt"

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

    pl_module = HandClassifierPL.load_from_checkpoint(checkpoint_path, model=model)

    pl_module.model.eval()

    prediction_logits = pl_module.model(img)

    prediction = int(torch.argmax(prediction_logits))

    print(__LABELS_STR__[prediction])
