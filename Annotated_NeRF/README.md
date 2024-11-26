# The Annotated NeRF: Training on Custom Dataset from Scratch in Pytorch

This folder contains the Jupyter Notebooks and scripts for the LearnOpenCV article  - **[The Annotated NeRF: Training on Custom Dataset from Scratch in Pytorch](https://learnopencv.com/annotated-nerf-pytorch)**.

<img src="media/NeRF_featured_gif.gif">

After downloading the dataset from the subscribe code button, perform the following steps,

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
After the end of training the model weights will be stored in the <expname>/logs folder. To do inference and extract mesh from the model, use the extract_mesh.ipynb notebook.

### Download datafrom here:
https://www.dropbox.com/scl/fi/ijhlr5n5gevf14ujijc2k/images_fps2.zip?rlkey=ruqy7op8olvfab6lrbodk5kxs&st=77qz1fsl&dl=1