The complete step-by-step procedure to execute 2D Gaussian Splatting on MipNeRF360 Flowers Dataset is given below. Each commands have to be executed on the terminal.

1. Start by cloning the 2DGS implementation (surfels-based)

.....
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
.....

2. Environment Setup

The repo provides an environment.yml file which Installs PyTorch + CUDA 11.8, includes Open3D and Trimesh for mesh extraction and visualization and builds CUDA extensions (diff-surfel-rasterization, simple-knn).

.....
conda env create --file environment.yml
conda activate surfel_splatting
.....

But during the creation of the environment, we might get a RuntimeError. The error occurs because there can be a mismatch between our system’s CUDA toolkit and the repo environment built with CUDA 11.8. To fix this, we can create a clean environment using PyTorch built for CUDA 12.1, which is compatible with your 12.2 runtime. Steps (terminal commands) to create a clean environment using PyTorch for CUDA 12.1 are given below –

.....
conda create -n surfel_splatting python=3.10 -y
conda activate surfel_splatting
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
.....

3. Install Remaining Dependencies

.....
conda install -c conda-forge ffmpeg=4.2.2 pillow=10.2.0 typing_extensions=4.9.0 pip=23.3.1
pip install open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4 scikit-image==0.21.0 tqdm==4.66.2 trimesh==4.3.2 plyfile opencv-python
.....

4. Build CUDA Submodules

.....
cd submodules/diff-surfel-rasterization
python setup.py install
 
cd ../simple-knn
python setup.py install
.....

5. Training Command

.....
python train.py -s /home/opencvuniv/Work/Shubham/2d-gaussian-splatting/360_extra_scenes/flowers -m output/m360/flowers
.....

This will load the images & camera poses, optimize surfel-based Gaussians and save the checkpoints to the path - output/m360/flowers/. Be careful about the path to the dataset to be given as an argument in the command.

6. Rendering Command

.....
python render.py -s /home/opencvuniv/Work/Shubham/2d-gaussian-splatting/360_extra_scenes/flowers -m output/m360/flowers --skip_train --skip_test --mesh_res 1024
.....

The above command loads your trained checkpoint, extracts the reconstructed base, and renders it with the extracted bounded mesh if we want to focus on the foreground.

.....
python render.py -s /home/opencvuniv/Work/Shubham/2d-gaussian-splatting/360_extra_scenes/flowers -m output/m360/flowers --unbounded --skip_train --skip_test --mesh_res 1024
.....

But if you want to render with unbounded mesh extraction, use the above terminal command for rendering.