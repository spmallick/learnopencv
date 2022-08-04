# Image Matting with state-of-the-art Method F, B, Alpha Matting

**This repository contains the code for [Image Matting with state-of-the-art Method F, B, Alpha Matting](https://www.learnopencv.com/image-matting-with-state-of-the-art-method-f-b-alpha-matting/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/sypd5up8yi8ysbe/AAB6iBRRYbIWMcqOozfkeTtEa?dl=1)

## Usage for test images

Please, follow the instruction to launch the demonstration script:

- Download model weights from [there](https://drive.google.com/uc?id=1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1);
- Install the requirements with `pip3 install -r requirements.txt`;
- Launch `python3 demo.py` to use default arguments. Use `python3 demo.py -h` for details.

The results will be saved into `./examples/predictions` by default.

## Usage for real life images

If you want to run the matting network on your own images you will need to generate the corresponding trimaps first. 
This process is supposed to be manual but that's too burdensome. Instead, please follow the instruction below to launch the trimap generation process using a semantic segmentation algorithm:

- You need to generate trimap. In this repo we use PyTorch implementation of the DeepLabV3 for that purpose. Select the class of your foreground object using - target class key:

```
python generate_trimaps.py -i /path/to/your/images (should be a directory) --target_class cat (consult --help for other options)
```

- Your trimaps will be stored into `path/to/your/images/trimaps` 
- Then launch
```
python demo.py --image_dir path/to/your/images --trimap_dir path/to/your/images/trimaps --output_dir path/to/save
```
  to get predictions
- Note that results may be imprecise due to rough trimap generation. You can try to play with the --conf_threshold to fix that.

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>