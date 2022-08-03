
# Face recognition with ArcFace

**This repository contains code for [Face recognition with ArcFace](https://www.learnopencv.com/face-recognition-with-arcface/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/peco9u3q485tems/AADaPVWIPn-Ly1LnuLKjHnmKa?dl=1)

## Original source code

Some parts of the code and trained face identification model are from [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) repository which is released under the [MIT License](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/LICENSE). Huge thanks to them!

## Installation

Use your Python virtual environment such as [virtualenv](https://virtualenv.pypa.io/en/latest/) to isolate project.

```
virtualenv -p python3.7 face-recognition
source face-recognition/bin/activate
```

Then install all dependencies.

```
pip install -r requirements.txt
```

_Note: GPU is not required to run this code, but model inference will be faster if you have one._

## Model
Download checkpoint for a model from [GoogleDrive](https://drive.google.com/drive/folders/1omzvXV_djVIW2A7I09DWMe9JR-9o_MYh)/[Baidu](https://pan.baidu.com/s/1L8yOF1oZf6JHfeY9iN59Mg#list/path=%2Fms1m-ir50) and move it to `checkpoint/backbone_ir50_ms1m_epoch120.pth`

## Data

All datasets with faces must support [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) format. Look at the prepared examples in `data` directory. For all subsequent commands use `tags` argument to select specific datasets in `data` directory.

## Data preprocessing
To prepare data with cropped and aligned faces from your original images, run:

```
python face_alignment.py --tags actors actresses musk woman --crop_size 112
```

_Note: crop_size argument must be either 112 or 224._

## Similarity visualization

To visualize similarity between faces in table format, run:

```
python similarity.py --tags actors actresses musk woman
```

The result for each dataset will be saved in `images` directory.

## t-SNE visualization

To use t-SNE for dimensionality reduction and 2D visualization of face embeddings, run:

```
python tsne.py --tags actors actresses musk woman
```

Results will be plotted in a separate window. You can enlarge the image to look at details.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

<a href="https://opencv.org/courses/">
<p align="center">
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>