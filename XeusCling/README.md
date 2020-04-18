# Xeus-Cling: Run C++ code in Jupyter Notebook

The repository contains notebook for **Facial Landmark Detection** demo in C++ with **C++11** kernel and **`includeLibraries.h`** header file for loading necessarily libraries. The blog post for the same can be found [here](https://www.learnopencv.com/xeus-cling-run-c-code-in-jupyter-notebook/).

## Installing Xeus-Cling

Create a new conda environment and activate it.

```
conda create -n xeus-cling
source activate xeus-cling
```

Use **conda package installer** to install Xeus-Cling.

```
conda install -c conda-forge xeus-cling
```

To create a Jupyter Notebook with C++ kernel, use the following.

```
source activate xeus-cling
jupyter-notebook
```
