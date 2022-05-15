from io import BytesIO
import base64

import numpy as np
import torch
from PIL import Image
from einops import rearrange


def convert_image(img_base64: str) -> torch.Tensor:
    # Convert from Base 64 to numpy img
    pil_img = Image.open(BytesIO(base64.b64decode(img_base64)))
    np_img = np.array(pil_img)

    # Get image information
    height, width, num_channels = np_img.shape

    # Get rid of the alpha channel if needed
    if num_channels == 4:
        np_img = np_img[:, :, :3]

    # Make the aspect ratio 1:1 by cropping
    m = min(height, width)

    h = (height - m) // 2
    w = (width - m) // 2

    np_img = np_img[h: h + m, w: w + m]

    # Scale the image to the 300x300 resolution
    pil_img = Image.fromarray(np_img)

    size = [300, 300]
    pil_scaled_img = pil_img.resize(size, Image.Resampling.LANCZOS)

    np_scaled_img = np.array(pil_scaled_img)

    # Convert the image to torch tensor
    torch_img = torch.from_numpy(np_scaled_img)

    # Add batch dim and move chanel dim in front
    torch_img = rearrange(torch_img, "h w c -> 1 c h w")

    # Convert from int to float
    torch_img = torch_img.float()
    torch_img /= 255

    return torch_img