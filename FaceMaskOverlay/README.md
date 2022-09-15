# Using Facial Landmarks for Overlaying Faces with Medical Masks

**This repository contains the code for [Using Facial Landmarks for Overlaying Faces with Masks](https://www.learnopencv.com/using-facial-landmarks-for-overlaying-faces-with-masks/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/zriu09eaunjp0lu/AADXO9ccFm1ELWvNh2YWqQAJa?dl=1)

Most of the code is based on the [HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) repository, huge thanks to  them.

## Quick start

### Environment

This code is developed using Python 3.6 and PyTorch 1.0.0 on Ubuntu 16.04 with NVIDIA GPU. Other platforms or GPUs are not fully tested.

### Installation

1. Create virtual environment:

You'll need to install [virtualenv](https://pypi.org/project/virtualenv/) package if you don't have it:

```bash
pip install virtualenv
virtualenv -p python3.6 venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. You will also need the pre-trained weights, which you can take from here
   [HR18-300W.pth](https://1drv.ms/u/s!AiWjZ1LamlxzeYLmza1XU-4WhnQ):

```bash
wget -O HR18-300W.pth https://pscm7q.by.files.1drv.com/y4m1ndEsUHxWtszPoyHY2BQ2Zdvh0-dgYW_5dtTcxX_YFP8p5YYADNSndm3tAj2f-U4aMPMuS6-VyMvWaCYaO2otLab4XWblhouZkbuIgzr3ZGem6A2b1Lm6Kb3WrYQL_m3D2hj8Y3ulD06kXpvsvsoN-YlmXd9NK12snBfQxrgQf7OVXYsP1xWJEZfN_1CKdLPl1xYNaNvCeQik5LiCnmB9g
```

### Run Demo

To run the demo, you need to execute the following command in your terminal:

```bash
python3 overlay_with_mask.py --cfg experiments/300w/face_alignment_300w_hrnet_w18.yaml --landmark_model HR18-300W.pth --mask_image masks/anti_covid.png
```

Try out different masks from the `masks` folder or use yours, but don't forget to annotate them first.

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>