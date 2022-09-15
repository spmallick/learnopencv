
# Real-time style transfer in a zoom meeting

**This folder contains code for [Real-time style transfer in a zoom meeting](https://learnopencv.com/real-time-style-transfer-in-a-zoom-meeting/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/rrwfezl8yt6q8uq/AACIi8im7TMYMiSxhHbSigWia?dl=1)

## Expected environment

Please use conda for setting up environment for this project conda env create -f env.yml.

In general, you need pytorch, opencv and pillow. CUDA acceleration is highly recommended for training, but inference can be done without it.

## Pretrained resnet18 at 640x480
We provide a pretrained resnet18 model file at 640x480 resolution. This is the most common resolution for webcams, so you can use this model as loss function for training style transfer models for webcams.

In case you still want to train resnet at another resolution, download imagenet data and create a text file containing paths of all images.

Set the path to the text file in `config.py`.

We use knowledge distillation for creating targets for training images.

```Shell
python3 precompute_targets.py
#After precomputing targets, we train

python3 train_resnet.py
```

You should see loss start to go down. You can also visualize the loss with tensorboard

```Shell
tensorboard --logdir=./runs/
```
## StyleNet training

The trained model will be saved to disk. Set the path of trained resnet model you want to use as LOSS_NET_PATH in `config.py`.

Set the path of any image you want to use as style target (STYLE_TARGET).

Train style transfer network with

```Shell
python3 stylenet.py
```

## Running live demo

After training, create virtual camera as explained in the blog post.

If you are on Windows/Mac, use

```Shell
python3 livedemo_macwin.py
```

If you are on linux, use

```Shell
python3 livedemo.py
```

Once the script is running, you can join any zoom/skype/teams meeting and choose the virtual camera. You will see stylized output and so will your friends in the meeting.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

<a href="https://opencv.org/courses/">
<p align="center">
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
