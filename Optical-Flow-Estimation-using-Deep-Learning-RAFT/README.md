
## RAFT: Optical Flow Estimation Using Deep Learning

**This repository contains code for [RAFT: Optical Flow estimation using Deep Learning](https://learnopencv.com/optical-flow-using-deep-learning-raft) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/p5kzzonq1wongjd/AADHW2hGKtpL4PVjmPAc4vCTa?dl=1)

## Installation

1. To run the demo, firstly you need to clone the [RAFT repo](https://github.com/princeton-vl/RAFT) being in our directory:

   ```Shell
   git clone git@github.com:princeton-vl/RAFT.git
   ```

   or

   ```Shell
   git clone https://github.com/princeton-vl/RAFT.git
   ```

   **Please, attention!** There is an option that authors can update their repo and our script will become non-working.
   To avoid this case, we saved the suitable version of the RAFT architecture in 
   [our GitHub](https://github.com/MaximKuklin/RAFT), so you can download it from there. 

2. To run the demo, you need to create a virtual environment your working directory:

   ```Shell
   virtualenv -p python3.7 venv
   source venv/bin/activate
   ```

   and install the required libraries:

   ```
   pip install -r requirements.txt
   ```

3. (Optional) There is a pretrained weights file that is already in our repo, but you can download all authors' weights files using this command:

   ```
   ./RAFT/download_models.sh
   ```

4. Now you can run the demo with RAFT:
   
   ```
   python3 inference.py --model=./models/raft-sintel.pth --video ./videos/crowd.mp4
   ```
   
   or with RAFT-S
   ```
   python3 inference.py --model=./models/raft-small.pth --video ./videos/crowd.mp4 --small
   ```
## Troubleshooting

If you have two GPUs and there is a User Warning like:

```Shell
UserWarning:
    There is an imbalance between your GPUs. You may want to exclude GPU 1 which
    has less than 75% of the memory or cores of GPU 0. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable.
```

with the following error such as:

```Shell
TypeError: forward() missing 2 required positional arguments: 'image1' and 'image2'
```

one of the solution is to set the environment variable `CUDA_VISIBLE_DEVICES` on our own:

```
$ export CUDA_VISIBLE_DEVICES=0
```

where `0` is the id number of the one of your GPUs.

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>