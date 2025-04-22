# MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors

This folder contains the instruction to run for the LearnOpenCV article  - **[MASt3R-SLAM Explanation](https://learnopencv.com/mast3r-slam-realtime-dense-slam-explained/)**.

<img src="readme_images/feature.gif">

To run:

## **MASt3R-SLAM**

Visit MASt3R-SLAM Repository 

```python
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# if you've clone the repo without --recursive run
# git submodule update --init --recursive
```

```python
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
 

# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1
```

**Download checkpoints**:

```mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/```
```

**Inference on a Image sequence**

````bash ./scripts/download_tum.sh
bash ./scripts/download_tum.sh
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml	
````

**Inference on a Video**

It can process either MP4 videos or folders containing RGB images.

```
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml
```

---

## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)