# Installing `gsplat` on Windows

## Install using a pre-compiled wheels 

You can install gsplat from python wheels containing pre-compiled binaries for a specific pytorch and cuda version. These wheels are stored in the github releases and can be found using simple index pages under https://docs.gsplat.studio/whl. 
You obtain the wheel from this simple index page for a specific pytorch an and cuda version by appending these the version number after a + sign (part referred a *local version*). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can use
```
pip install gsplat==1.2.0+pt20cu118 --index-url https://docs.gsplat.studio/whl
```
Alternatively, you can specify the pytorch and cuda version in the index url using for example
```
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```
This has the advantage that you do not have to pin a specific version of the package and as a result get automatically the latest package version.


## Install from source

You can install gsplat by compiling the wheel. In this way it will build the CUDA code during installation. This can be done using either the source package from pypi.org the wheel from pypi.org or using a clone of the repository. In all case Visual Studio needs to be install and activated.

### Visual studio setup

Setting up and activating Visual Studio can be done through these steps:

1. Install Visual Studio Build Tools. If MSVC 143 does not work, you may also need to install MSVC 142 for Visual Studio 2019. And your CUDA environment should be set up properly.


2. Activate your conda environment:
    ```bash
    conda activate <your_conda_environment>
    ```
    Replace `<your_conda_environment>` with the name of your conda environment. For example:
    ```bash
    conda activate gsplat
    ```

3. Activate your Visual C++ environment:
    Navigate to the directory where `vcvars64.bat` is located. This path might vary depending on your installation. A common path is:
    ```
    C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build
    ```

4. Run the following command:
    ```bash
    ./vcvars64.bat
    ```

    If the above command does not work, try activating an older version of VC:
    ```bash
    ./vcvarsall.bat x64 -vcvars_ver=<your_VC++_compiler_toolset_version>
    ```
    Replace `<your_VC++_compiler_toolset_version>` with the version of your VC++ compiler toolset. The version number should appear in the same folder.
    
    For example:
    ```bash
    ./vcvarsall.bat x64 -vcvars_ver=14.29
    ```

### Install using the source package published on `pypi.org`

You can install gsplat from the published source package (and not the wheel) by activating Visual Studio (see above) and then using
```
pip install --no-binary=gsplat gsplat --no-cache-dir
```
The CUDA code will be compiled during the installation and the Visual Studio compiler `cl.exe` does not need to be added to the path, because the installation process as an automatic way to find it.
We use `--no-cache-dir` to avoid the potential risk of getting pip using a wheel file from `pypi.org` that would have be downloaded previously and that does not have the binaries.

### Install using the wheel published on `pypi.org`

Setting up and activating Visual Studio can be done through these steps:
You can install the `gsplat` using the wheel published on `pypi.org` by activating Visual Studio (see above) and then using 
```
pip install gsplat
```
The wheel that does not contain the compiled CUDA binaries. The CUDA code is not compiled during the installation when using wheels and will be compiled at the first import of `gsplat` which requires the Visual Studio executable `cl.exe` to be on the path (see pre-requisite section above). 

### Install using the clone of the Repository
This can be done through these steps:
1. Clone the `gsplat` repository:
    ```bash
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git
    ```

2. Change into the `gsplat` directory:
    ```bash
    cd gsplat
    ```
3. Activate visual Studio
4. Install `gsplat` using pip:
    ```bash
    pip install .
    ```
