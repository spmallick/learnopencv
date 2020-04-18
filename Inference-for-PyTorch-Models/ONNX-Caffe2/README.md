# PyTorch Model Inference using ONNX and Caffe2

We show how to do inference with a PyTorch model using ONNX and Caffe2.

## Dependencies

torch  
torchvision  
onnx

## Create conda environment

```conda env create -f environment.yml```

## Download models
The required models can be downloaded from the following links:
[Animals Caltech](https://drive.google.com/open?id=14XvkumHXxGWed_osX_XpBRLOVA6v9WHl)

## Activate conda environment

```conda activate pytorch_inference```

## Outline
The enclosed notebook lets you
1. Export the PyTorch .pt model to ONNX model
2. Use the ONNX model for inference in Caffe2

More details can be found in the [blog](https://www.learnopencv.com/pytorch-model-inference-using-onnx-and-caffe2/)
