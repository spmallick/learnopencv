# Multi Object Tracking using OpenCV
This repository contains the code for [MultiTracker : Multiple Object Tracking using OpenCV (C++/Python)](https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/) blog.

## Instructions for C++ code
1. Download the code and extract the downloaded file, if required.
2. Create a `build` folder: `mkdir build && cd build`.
3. System specific instructions:
    1. **For Windows Operating System**: Tell CMake to configure and generate project files in Visual Studio 14 x64 format.
Use `..` to look for CMakeLists.txt file in the parent folder of build directory.
It will generate project files in Visual Studio 14’s format.
`cmake -G "Visual Studio 14 2015 Win64" ..`
    2. **For Linux/MacOS**:
Tell CMake to configure and generate project files.
`..` denotes that CMake should look for CMakeLists.txt file in the parent folder of build directory.
For Linux/MacOS cmake detects and uses the C++ toolchain installed on your system
It will most likely be gcc toolchain.
`cmake ..`
4. Now tell CMake to build project in Release mode: `cmake --build . --config Release`
5. When the build is completed, executables will be generated in `build\Release` folder or `build` folder itself.
6. Now go back to project directory (parent directory of build) and run the compiled script: `./build/multiTracker`

## Instructions for Python code
1. Download the code and extract the downloaded file, if required.
2. Use `python multiTracker.py` to run the code.

## Tracker selection
The scripts use `CSRT` as the default tracking algorithm. To use any other algorithm, change `trackerType` variable to the tracker of your choice.

Note that only these tracking algorithms are available in the scripts: `BOOSTING`, `MIL`, `KCF`, `TLD`, `MEDIANFLOW`, `GOTURN`, `MOSSE`, `CSRT`.

To change the `trackerType` variable in C++, go to [line number 58](https://github.com/spmallick/learnopencv/blob/master/MultiObjectTracker/multiTracker.cpp#L58) and change the tracker to `"BOOSTING"`, for example.

To do the same for Python code, edit `trackerType` variable in [line number 46](https://github.com/spmallick/learnopencv/blob/master/MultiObjectTracker/multiTracker.py#L46).

## GOTURN Tracker
For instructions regarding how to use GOTURN tracker, please refer to the blog on [GOTURN: Deep Learning based Object Tracking](https://www.learnopencv.com/goturn-deep-learning-based-object-tracking) and the corresponding [GitHub repository](https://github.com/spmallick/learnopencv/tree/master/GOTURN).
