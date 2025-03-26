# MASt3R and MASt3R-SfM Explanation: Image Matching and 3D Reconstruction Results

This folder contains the Jupyter Notebook for the LearnOpenCV article  - **[MASt3R and MASt3R-SfM Explanation](https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/)**.

<img src="readme_images/feature.gif">

**To download the code hit the below button**:

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download" width="200">](https://www.dropbox.com/scl/fo/fbihe4pjqgi9ki9nk5li8/AAKQOTmpsws9Km99WyZF4RY?rlkey=9aecv2v4z59z2xcu2ghtx6ybs&st=hjb7wlcf&dl=1)

To run:

## **MASt3R**

Visit MASt3R Repository : [Link](https://github.com/naver/mast3r)

```python
git clone --recursive https://github.com/naver/mast3r
cd mast3r
# if you have already cloned mast3r:
# git submodule update --init --recursive
```

```python
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add required packages for visloc.py
pip install -r dust3r/requirements_optional.txt
```

**Download checkpoints**:
`mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/`

**Gradio Demo**

`python3 demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`	

The input can be a single image or multiple images, and if you have limited GPU Resources say 6GB or 12 GB, for larger scenes go with `one-ref` pairing strategy. Otherwise use complete-pairing strategy for small subsets.

---

## **MAST3R-SfM**

Visit MASt3R-SfM Branch of MASt3R Repository : [Link](https://github.com/naver/mast3r/tree/mast3r_sfm)

`git clone -b mast3r_sfm https://github.com/naver/mast3r `

```
pip install cython
git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .  # or python3 setup.py build_ext --inplace
cd ..
```

```
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

To use MASt3R for retrieval to construct the Sparse Scene Graph

```
python3 demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --retrieval_model checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth

# Use -
# Use --retrieval_model and point to the retrieval checkpoint (*trainingfree.pth) to enable retrieval as a pairing strategy, asmk must be installed
```

##### Gradio Configuration to use `retrieval: connect views based on similarity`

![img](https://learnopencv.com/wp-content/uploads/2025/03/mast3r-sfm-configurations-scene-graph-retrievalconnect-views-from-graph_-1024x431.png)

## INSTANTSPLAT

Clone this Fork: 	

`https://github.com/jonstephens85/InstantSplat_Windows`

Modify `instantsplat_gradio.py`:

```
# instantsplat_gradio.py 
n_views = gr.Dropdown(choices=[3, 6, 12], value=3, label="Number of Views")
 #     (to)
n_views = gr.TextBox(label="Total images") # len(images)
```

Then run,

`!python instantsplat_gradio.py`

For more details: [InstantSplat Tutorial](https://www.youtube.com/watch?v=VHDq2v8hEA8)

**Datasets used in Article**:

InstantSplat Preprocessed Dataset: [Download Link](https://drive.google.com/file/d/1Z17tIgufz7-eZ-W0md_jUlxq89CD1e5s/view)

1. [Tanks and Temples](https://www.tanksandtemples.org/download/) - Family | Church    >>> Image Set
2. [CO3D](https://ai.meta.com/datasets/co3d-downloads/)   - motorcycle | teddybear
3. DSLR Camera: [ Download Link ](https://www.dropbox.com/scl/fi/ijhlr5n5gevf14ujijc2k/images_fps2.zip?rlkey=ruqy7op8olvfab6lrbodk5kxs&st=77qz1fsl&dl=1)



---

## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)