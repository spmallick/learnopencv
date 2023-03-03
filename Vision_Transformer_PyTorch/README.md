# Vision Transformer PyTorch

Implementation of Vision Transformer in PyTorch

## Files
- `mhsa.py`: Implementation of Multi Head Self Attention layer
- `vitconfigs.py`: Configs for base (ViT-B), large (ViT-L) and huge (ViT-H) models as described by [Dosovitskiy et. al.](https://arxiv.org/abs/2010.11929)
- `vit.py`: Implementation of Vision Transformer
- `train.py`: Training script for ViT on imagenet dataset using [DarkLight](https://github.com/dataplayer12/darklight)

## Environment
Set up an environment with pytorch and TensorRT. The easiest way is to use an NGC container like this (note that a CUDA GPU is required for training):

```Shell
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.01-py3
```

## Verify ViT forward pass
```Shell
python3 vit.py #this will print a verification message if fwd pass is successful.
```

## Training
In the docker container, load an external volume which contains imagenet dataset. The dataset should have the format:
```Shell
- root
	|
	|
	|--- train
	|		|
	|		|_ timg1.jpg
	|		|_ timg2.jpg
	|		...
	|
	|--- val
	|		|
	|		|_ vimg1.jpg
	|		|_ vimg2.jpg
	|		...
```
The image names contain the class label in imagenet. 
Provide the path of the root dir in `train.py`

Run training with
```Shell
python3 train.py
```
Visualize training progress with tensorboard.
```Shell
tensorboard --logdir=./runs
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://camo.githubusercontent.com/5c10c2db6c1c005a3846ca4e1774a650346ef7e0be436aa7b39e50210d2a80af/68747470733a2f2f6c6561726e6f70656e63762e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032332f30312f41492d436f75727365732d42792d4f70656e43562d4769746875622e706e67)](https://opencv.org/courses/)
