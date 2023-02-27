# Build a Video to Slides Converter Application using the Power of Background Estimation and Frame Differencing in OpenCV

**This repository contains code for [Build a Video to Slides Converter Application using the Power of Background Estimation and Frame Differencing in OpenCV](https://learnopencv.com/video-to-slides-converter-using-background-subtraction/) blogpost**


## Video to Slides Conversion using Frame Differencing and Background Estimation

<br>

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/scl/fo/3ln9cx3ddrmfiapa9atr5/h?dl=1&rlkey=j61webbj820sfwuggo58fgsd5)



### Install required packages

After unzipping the file `Build-a-Video-to-Slides-Converter-Application-using-the-Power-of-Background-Estimation-and-Frame-Differencing-in-OpenCV`, run the following command in your virtual environment:
```
pip install -r requirements.txt
```

### Execution

The command line flags are as follows:

* `video_file_path`: The path to the input video file.
* `out_dir`: The path to the output directory where the results would be stored.
* `type`: The type of background subtraction method to be applied. It can be one of: Frame_Diff, GMG (default), or KNN.
* `no_post_process`: flag to specify whether to apply the post-processing step. If not specified, the post-processing step is always applied as default.
* `convert_to_pdf`: flag to specify whether to convert the image set into a single PDF file.


An example use case can be:

```
python video_2_slides.py -v ./samples/vid_test.mp4 -o output_results --type GMG --convert_to_pdf
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
