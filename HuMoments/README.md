
# Shape Matching using Hu Moments

This directory contains code for using Hu Moments to match shapes.

# For C++

## How to compile the code
If you don't have OpenCV installed globally, then Specify the **OpenCV_DIR** in CMakeLists.txt file. Then,

```
cmake .
make
```
# How to Run the code

## C++ ##
## Find Hu Moments
```
./HuMoments images/*
```

## Match shapes
```
./shapeMatcher
```


## Python ##
## Find Hu Moments
```
python HuMoments.py images/*
```

## Match shapes
```
python shapeMatcher.py
```
