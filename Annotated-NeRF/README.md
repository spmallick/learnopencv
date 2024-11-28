# The Annotated NeRF: Training on Custom Dataset from Scratch in Pytorch

This folder contains the Jupyter Notebooks and scripts for the LearnOpenCV article  - **[The Annotated NeRF: Training on Custom Dataset from Scratch in Pytorch](https://learnopencv.com/annotated-nerf-pytorch)**.

<img src="media/NeRF_featured_gif.gif">


## Run on Google Colab
<a href="https://colab.research.google.com/drive/146xc4HcoiVCD_JTsefi9GYk5kXvjR4Cs?usp=sharing">
        <img alt="colab" src="https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/docs/_static/imgs/readme_colab.png" width="150"></a>

## Download datafrom here:
- [DSLR dataset](https://www.dropbox.com/scl/fi/ijhlr5n5gevf14ujijc2k/images_fps2.zip?rlkey=ruqy7op8olvfab6lrbodk5kxs&st=77qz1fsl&dl=1)

- [Dataset Link](https://www.dropbox.com/scl/fo/5tm5p4ftky14pr1amzidq/AEK-tyOrtvc3a_ydfOgbreI?rlkey=fjdxfphoaods07ame271rc6kh&st=x1glssoj&dl=1)


## Model Training Locally

After downloading the dataset, perform the following steps,

This command converts the video into frames and stores them inside the output_dir.

```
$ python video2imgs.py --video_path captain_america_v1.mp4 --output_dir /path/to/dataset --fps 5
```

Now, we will run COLMAP and convert the data into llff format. factor can be anything (generally 2-8 based on the original image size). The value of this factor parameter also needs to be updated in the config.txt file.

```
$ python imgs2poses.py --data_dir "/path/to/dataset" --factor 4
```

After preparing the dataset, we will update the config file based on the prepared dataset directory, and factor parameter etc. After that's done we can directly run below command to start training,

```
$ python run_nerf.py --config configs/<dataset_name>.txt
```
After training you can do inference using the below command, it will generate for both disparity map as well as the 360 degree rendered video.

```
$ python run_nerf.py --config configs/<dataset_name>.txt --render_only
```

After the end of training the model weights will be stored in the <expname>/logs folder. To do inference and extract mesh from the model, use the extract_mesh.ipynb notebook.
