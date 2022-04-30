import os
from typing import Literal

import cv2 as cv
import torch
from einops import rearrange
from torch.utils.data import TensorDataset

from data_manager.code.download_dataset import __STORAGE_RAW_PATH__, get_data_local

__IMAGE_SIZE__ = 300
__IMAGE_SHAPE__ = (3, __IMAGE_SIZE__, __IMAGE_SIZE__)

__SUB_DIR__ = "rps"

__LABELS_STR__ = ["rock", "paper", "scissors"]


def rgb_read_image(img_path: str):
    bgr_img_np = cv.imread(img_path)
    b, g, r = cv.split(bgr_img_np)
    rgb_image_np = cv.merge([r, g, b])
    rgb_image_torch = torch.from_numpy(rgb_image_np)
    return rearrange(rgb_image_torch, "w h c -> 1 c w h")


def load_dataset(split: Literal["train", "test"], verbose: bool = False):
    get_data_local(verbose)

    path = os.path.join(__STORAGE_RAW_PATH__, split, __SUB_DIR__)

    targets = []
    images = []

    for i, img_label in enumerate(__LABELS_STR__):
        img_label_path = os.path.join(path, img_label)

        file_names = os.listdir(img_label_path)
        for file_name in file_names:
            file_path = os.path.join(img_label_path, file_name)
            image_tensor = rgb_read_image(file_path)
            images.append(image_tensor)

        targets.extend([i] * len(file_names))

    images = torch.vstack(images)
    images = images.float()
    images /= 255
    targets = torch.tensor(targets)

    return TensorDataset(images, targets)


def get_image_shape():
    return __IMAGE_SHAPE__
