# Invisibility Cloak
Create your own invisibility cloak using OpenCV

## Download the input video

The input video can be downloaded using this link: https://drive.google.com/file/d/1rc13wZ9zC03ObG5zB3uccUtsg_rsI8hC/view?usp=sharing

## Using the C++ code

### Compilation

To compile the **`Invisibility_Cloak.cpp`** code, use the following:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Usage

Refer to the following to use the compiled file:

```
./build/Invisibility_Cloak --video=Input.mp4
```

To take input from camera, use:

```
./build/Invisibility_Cloak
```

## Using the Python code

### Usage

Refer to the following to use the Python script:

```
python Invisibility_Cloak.py --video Input.mp4
```

To take input from camera, use:

```
python Invisibility_Cloak.py
```


