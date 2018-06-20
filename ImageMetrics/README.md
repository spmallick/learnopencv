Please see the following [blog post](<link_goes_here>) for more details about this code

[Image Quality Assessment: BRISQUE, a No-Reference Metric](<link_goes_here>)

## Installation Instructions
**Python 2.x LIBSVM Installation**
`sudo apt-get install python-libsvm`

**Python 3.x LIVSVM Installation and C++ LIBSVM Installation**
For C++ : 
`cd C++/libsvm/`
`cmake .`
`make`

For Python 3.x :
`cd Python/libsvm/`
`make`
`cd python`
`make`

## Usage 
**Python 2.x**
python2 brisque_final.py <image_path>

**Python 3.x** 
cd Python/libsvm/python/
python3 brisque_final.py <image_path>

**C++**
`cd C++/`
`./brisquerevised -im <image_path>`