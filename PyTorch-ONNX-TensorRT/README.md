# How to convert a model from PyTorch to TensorRT and speed up inference
The blog post is here: https://www.learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/

To run Python part:
```shell script
python3 -m pip install -r requirements.txt
python3 pytorch_model.py
```

To run C++ part:
```shell script
mkdir build
cd build
cmake -DOpenCV_DIR=[path-to-opencv-build] -DTensorRT_DIR=[path-to-tensorrt] ..
make -j8
trt_sample[.exe] resnet50.onnx turkish_coffee.jpg
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>