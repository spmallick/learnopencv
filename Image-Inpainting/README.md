# Image Inpainting


## Usage

### Python

```
python3 inpaint.py sample.jpeg
```

### C++

```
g++ inpaint.cpp `pkg-config opencv --cflags --libs` -o inpaint
./inpaint sample.jpeg
```
You can also **cmake** as follows:

```
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

The built code can then be used as follows:

```
./build/inpaint sample.jpeg
```

## Performance Comparison

```
Time: FMM = 194445.94073295593 ms
Time: NS = 179731.82344436646 ms
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
