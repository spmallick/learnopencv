# Photoshop Filters in OpenCV

## Filters Available

The following filters are available in the repository:

1. Brightness
2. 60s TV
3. Emboss
4. Duo tone
5. Sepia

## Instructions

### Python

In order to run the specific filter, please use `python <filter-name>.py`, for example, `python duo_tone.py`

### C++

1. Compile all the code files using the following steps:

```mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
```

2. To run the specific filter, please use `./filter-name`, for example, `./duo_tone`
