from torch import nn
from torchvision.transforms import RandomHorizontalFlip, RandomPerspective, RandomResizedCrop, RandomErasing, RandomAffine, \
    RandomRotation, ColorJitter


def create_data_augmenter() -> nn.Module:
    return nn.Sequential(
        RandomHorizontalFlip(),
        RandomPerspective(fill=1),
        RandomResizedCrop(size=300, scale=(0.90, 1.00)),
        RandomErasing(value=1, scale=(0.1, 0.15)),
        RandomAffine(degrees=30, fill=1),
        RandomRotation(degrees=30, fill=1),
        ColorJitter(brightness=(0.8, 1.0)),
        ColorJitter(contrast=(0.8, 1.0)),
        ColorJitter(saturation=(0.5, 1.0)),
    )