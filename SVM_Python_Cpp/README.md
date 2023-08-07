# SVM using Python and C++

## Requirements

In order to run the Python scripts, you will need to install `scikit-learn` module using `pip install scikit-learn`

For running the C++ code, make sure you download and place all the files provided in the repository following the same folder structure.

## Compilation Instructions

To run the Python scripts, use:

```
python SVM_Regression.py
python SVM_Classification.py
```

To run the C++ code, use:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
```

The executable files will be created in the folder `bin/`. You can run regression example using `svm_regression` binary and classification example using `svm_classification` binary.
