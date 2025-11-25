The complete step-by-step procedure to execute Image-GS Image Reconstruction pipeline using 2D Gaussians is given below. Each commands have to be executed on the terminal.

1. First, clone the Image-GS Repository by running the following command in Shell:

.....
git clone https://github.com/NYU-ICL/image-gs.git
cd image-gs
.....

2. Create the conda environment from environment.yml

.....
conda env create -f environment.yml
conda activate image-gs
.....

3. Include fused-ssim

.....
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
.....

4. Install gsplat (Correct Version, Correct Build Flags). Two things matter here:
   We must avoid pip’s “build isolation” (or the build won’t see torch).
   We must use a compatible gsplat version (1.3.x), because 1.4.0 changed the API.
   --no-build-isolation → prevents pip from creating a temporary, torch-less build environment.
   --no-deps → avoids pip trying to “helpfully” reinstall a different torch.

.....
cd gsplat
 
# Make sure we use this env's Python
pip install --upgrade pip setuptools wheel
 
# Install a compatible gsplat version
pip install gsplat==1.3.0 --no-build-isolation --no-deps
.....

5. Install the Image-GS Package Itself while still using the same torch & gsplat we’ve already set up.

.....
cd ..
python -m pip install -e . --no-build-isolation --no-deps
.....

6. The Python script (model.py) has to be replaced with the original model.py Python script, which was cloned into the system while cloning the Image-GS repository. Because of some updates to the file structure, a few imports are wrong, along with some script names, which is why the replacement of the model.py Python script is necessary. The concerned Python script, along with all the instructions, can be downloaded by clicking the Download Code button, right before the "Implementing the Image-GS Pipeline in Practice" section of the blog post.
Once everything is installed, running Image-GS on an example image is straightforward.

7. Download the image and texture datasets from OneDrive (https://1drv.ms/u/c/3a8968df8a027819/EeshjZJlMtdCmvvmESiN2pABM71EDaoLYmEwuOvecg0tAA?e=GybqBv) and organize the folder structure as follows –
image-gs
└── media
    ├── images
    └── textures

8. Shell command to reconstruct the input image

.....
python main.py --input_path="images/art-5_2k.png" --exp_name="test/art-5_2k" --num_gaussians=10000 --quantize
.....

9. Shell command to render the upsampled image with size 4k x 4k 

.....
python main.py --input_path="images/art-5_2k.png" --exp_name="test/art-5_2k" --num_gaussians=10000 --quantize --eval --render_height=4000
.....