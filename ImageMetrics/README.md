Please see the following [blog post](https://www.learnopencv.com/image-quality-assessment-brisque/) for more details about this code

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
`python2 brisquequality.py <image_path>`

**Python 3.x** 
`cd Python/libsvm/python/`

`python3 brisquequality.py <image_path>`

**C++**
`cd C++/`

`./brisquequality <image_path>`
