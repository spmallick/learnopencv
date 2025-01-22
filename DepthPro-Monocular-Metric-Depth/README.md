# Depth Pro: Sharp Monocular Metric Depth - Paper Explanation and Applications

This folder contains the Jupyter Notebooks and Scripts for the LearnOpenCV article  - **[Apple Depth Pro: Sharp Monocular Metric Depth](https://learnopencv.com/depth-pro-monocular-metric-depth/)**.

<img src="readme_images/Feature.gif">


### To run:
```python
!git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
```

```python
#setup
!pip install -e .
```

You will need to download the pre-trained checkpoint using the following bash command which will place the model under `ml-depth-pro/checkpoints/depth-pro.pt` folder.

**Download checkpoints**:
`source get_pretrained_models.sh`

**Image Inference – Usage**

```
!depth-pro-run -i image.jpg -o output_dir
```

The input can be an image or a directory containing multiple images, the `output_dir` will store the resulting inverse depth maps.

We have modified the `ml-depth-pro/src/depth_pro/cli/run.py`  to save raw depth and surface normal and is named as `depthpro-app.py`

---

To run,
`python depthpro-app.py`

To visualize metric depth per each pixel with a interactive OpenCV window, run 

`python metric-depth-visualize.py`

---

### Scripts: Applications of Depth Data In Image Editing

- **Parallax Effect with Adobe Software** - `parallax-effect.py`
- **Depth of Field Effect** - `depth_of_field.py`
- **Depth Blur / Portrait** - `depth_blur.py`
- **3D Point Cloud Projection** - `depthToPointCloud.ipynb`

You can directly download the code files from the below link.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://www.dropbox.com/scl/fi/59iskkpdh4hqkv4axdmy9/DepthPro-Monocular-Metric-Depth.zip?rlkey=40pqu0nldvsfn4c1dd976ubtl&st=dmrlkjtx&dl=1)

## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)