# Building An Automated Image Annotation Tool: PyOpenAnnotate

This respository contains code for the blog post [Building An Automated Image Annotation Tool: PyOpenAnnotate](https://learnopencv.com/building-automated-image-annotation-tool-pyopenannotate).

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">]()

## Installation 

```
pip install pyOpenAnnotate
```

## How To Use pyOpenAnnotate?
Annotating images using pyOpenAnnotate is pretty simple. Use the command `annotate` followed by the following flags as per the requirement.

### 1. Annotate Images

```
annotate --img <images_directory_path>
```

### 2. Annotate Video
```
annotate --vid <path_to_video_file>
```
### 3. Global Flags
```
-T : View mask window.
--resume <existing-annotations-dir>: Continue from where you left off.
--skip <int(Frames)> : Frames to skip while processing a video file.
```

### 4. Mouse Controls
```
Click and Drag: Draw bounding boxes.
Double Click: Remove existing annotation.
```

## Display Annotations
Visualize your annotations using the `showlbls` command.
```
showlbls --img <single_image_or_a_directory> --ann <single_annotation_text_file_or_a_directory>
```

## Keyboard Navigation
```
N or D : Save and go to next image
B or A : Save and go back
C : Toggle clear screen (during annotation)
T : Toggle mask window (during annotation)
Q : Quit
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
