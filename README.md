# Rock Paper Scissors - Computer Vision Project

## Table of Contents

- [Description](#description)
- [Team Members](#team-members)
- [Model - Vision Transformer](#model---vision-transformer)
- [Dataset](#dataset)

## Description

The project proposes to create a model that is able to identify if a given image
of a hand is representing the sign Rock, Paper or Scissors.

This module is supposed to be used inside another project, to make a fully
functional online Rock Paper Scissors game, using just the webcams.

## Team Members

- Alexandru Știrbu (
    [LinkedIn](https://www.linkedin.com/in/alexandru-%C8%99tirbu-748068177/) | 
    [GitHub](https://github.com/Akrielz)
  )
- Alexandru Cojocariu ( LinkedIn | GitHub )
- Mihai Craciun ( 
    [LinkedIn](https://www.linkedin.com/in/craciun-m-3366aa122/) | 
    [GitHub](https://github.com/NiceDayZ)
  )

## Model - Vision Transformer

### Architecture Design

The model is a [Vision Transformer](https://openreview.net/pdf?id=YicbFdNTTy),
implemented in [PyTorch](https://pytorch.org/), following the template provided
by [Phil Wang (lucidrains)](https://github.com/lucidrains) on his 
[vit-pytorch](https://github.com/lucidrains/vit-pytorch#vision-transformer---pytorch) 
repository.

<img src="./readme_assets/vit.gif" width="500px"></img>

The Transformer is modified to have Self-Attend blocks, implemented accordingly
to [Perceiver AR paper](https://arxiv.org/abs/2202.07765), featuring residual
pre-normalized Multi Head Self Attention and Feed Forward blocks. 
The MLP blocks use Squared Relu as inner activation.

The Attention layer is also modified to apply [Rotary Embedding](https://arxiv.org/abs/2104.09864) 
for Keys and Queries if needed. This uses [lucidrains](https://github.com/lucidrains/rotary-embedding-torch)
implementation.

### Usage

```python
import torch
from vision_transformer import VisionTransformer

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
```

### Parameters

- `image_size`: int.  
Image size. If you have rectangular images, make sure your image size is the maximum of the width and height


- `patch_size`: int.  
Number of patches. `image_size` must be divisible by `patch_size`.  
The number of patches is: ` n = (image_size // patch_size) ** 2` and `n` **must be greater than 16**.


- `num_classes`: int.  
Number of classes to classify.


- `dim`: int.  
Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.


- `depth`: int.  
Number of Transformer blocks.


- `heads`: int.  
Number of heads in Multi-head Attention layer. 


- `mlp_dim`: int.  
Dimension of the MLP (FeedForward) layer. 


- `channels`: int, default `3`.  
Number of image's channels. 


- `dropout`: float between `[0, 1]`, default `0`.  
Dropout rate. 


- `emb_dropout`: float between `[0, 1]`, default `0`.   
Embedding dropout rate.


- `dim_head`: int, default to `64`.  
The dim for each head for Multi-Head Attention.


- `pool`: string, either `cls` or `mean`, default to `mean`  
Determines if token pooling or mean pooling is applied


- `apply_rotary_emb`: bool, default `False`.  
If enabled, applies rotary_embedding in Attention blocks.

## Dataset

### General

Used the [Rock Paper Scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) 
data provided by [TensorFlow](https://www.tensorflow.org/).

Because the Tensorflow API would return TensorFlow tensors, the project provides 
a script that downloads the zip files, extracts it into  splits, and returns a 
PyTorch TensorDataset for the desired split, containing the image tensors of 
shape `[<num_files>, 3, 300, 300]`, and targets of shape `[<num_files>]`.

The dataset is connected through [PyTorch Lightning](https://www.pytorchlightning.ai/)
modules to the model. A practical examples is available in `main.py`.

### Usage

```python
from data_manager import load_dataset

dataset = load_dataset(split="train", verbose=True)
```