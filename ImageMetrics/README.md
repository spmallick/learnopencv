Please see the following [blog post](https://www.learnopencv.com/image-quality-assessment-brisque/) for more details about this code

## Installation Instructions
**Python 2.x LIBSVM Installation**
`sudo apt-get install python-libsvm`

**Python 3.x LIVSVM Installation and C++ LIBSVM Installation**

For C++ :

1. `cd C++/libsvm/`
2. `cmake .`
3. `make`

For Python 3.x :

1. `cd Python/libsvm/`
2. `make`
3. `cd python`
4. `make`

## Usage 

**Python 2.x**

1. `python2 brisquequality.py <image_path>`

**Python 3.x** 

1. `cd Python/libsvm/python/`
2. `python3 brisquequality.py <image_path>`

**C++**

1. `cd C++/`
2. `./brisquequality <image_path>`
