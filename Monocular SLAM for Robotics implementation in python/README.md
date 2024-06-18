### Implementation of Monocular Visual SLAM in Python:


## Setup pangolin for python:

#### Install pangolin python:
The [original library](https://github.com/stevenlovegrove/Pangolin) is written in c++, but there is [python binding](https://github.com/uoip/pangolin) available. 

- **Install dependency:** For Ubuntu/Debian execute the below commands to install library dependencies,   

```
sudo apt-get install libglew-dev
sudo apt-get install cmake
sudo apt-get install ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev
sudo apt-get install libdc1394-22-dev libraw1394-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libopenexr-dev
```

- Don't need to follow the [Very Optional Dependencies](https://github.com/uoip/pangolin?tab=readme-ov-file#very-optional-dependencies) from the repository.

- **Install the Library:** Execute the below commands to install *pangolin*,
```
git clone https://github.com/uoip/pangolin.git
cd pangolin
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install
```

In the `make -j8` you might get some error, just follow the comment mentioned in this [github issue](https://github.com/uoip/pangolin/issues/33#issuecomment-717655495). Running the `python setup.py install` might throw an silly error, use this [comment](https://github.com/uoip/pangolin/issues/20#issuecomment-498211997) from the exact issue to solve this. 

- Other dependencies are pip installable.

 
## How to run?

```bash
python main.py
```

## Code structure:
```bash
├── display.py
├── extractor.py
├── pointmap.py
├── main.py
├── notebooks
│   ├── bundle_adjustment.ipynb
│   ├── mapping.ipynb
│   └── SLAM_pipeline_step_by_step.ipynb

```

In the notebook section we have shown how to run all the components of a monocular slam,
- `SLAM_pipeline_step_by_step.ipynb` Describes the entire pipeline
- `mapping.ipynb` is another resource for mapping [source](https://github.com/SiddhantNadkarni/Parallel_SFM)
-  `bundle_adjustment.ipynb` another great resource to understand g2o and bundle adjustment. [source](https://github.com/maxcrous/multiview_notebooks)

1st notebook uses the kitti dataset (grayscale, 22 GB), [download it from here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).