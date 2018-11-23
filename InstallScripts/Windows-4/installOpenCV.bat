::============================================::
set "cvVersion=master"
echo Installing OpenCV-%cvVersion%
::============================================::
mkdir opencv-%cvVersion%
cd opencv-%cvVersion%
set count=1
set "cwd=%cd%"
::============================================::
echo "Creating python environments"
::create python3 virtual environments
CALL conda create -y -f -n OpenCV-%cvVersion%-py3 python=3.6 anaconda
CALL conda install -y -n OpenCV-%cvVersion%-py3 numpy scipy matplotlib scikit-image scikit-learn ipython dlib
::============================================::
echo "Downloading opencv from github"
::download opencv from git
git clone https://github.com/opencv/opencv.git
cd opencv
::checkout appropriate cv version
git checkout %cvVersion%
cd ..
::download opencv_contrib from git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
::checkout appropriate opencv_contrib version
git checkout %cvVersion%
cd ..
::============================================::
CALL conda activate OpenCV-%cvVersion%-py3
::////////////////////////////////////////////::
FOR /F "tokens=* USEBACKQ" %%a IN (`where python`) DO (
	set var!count!=%%a
	set /a count=!count!+1
)
cd %var1%\..\..
::////////////////////////////////////////////::
set envsDir=%var1%\..\..
cd %cwd%
CALL deactivate
::============================================::
echo "Compiling using cmake"
cd opencv
mkdir build
cd build
::xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx::
cmake -G "Visual Studio 14 2015 Win64" -DINSTALL_C_EXAMPLES=ON -DINSTALL_C_EXAMPLES=ON -DINSTALL_PYTHON_EXAMPLES=ON -DOPENCV_EXTRA_MODULES_PATH=%cwd%\opencv_contrib\modules -DBUILD_opencv_saliency=OFF -DPYTHON3_EXECUTABLE=%envsDir%\OpenCV-%cvVersion%-py3\python.exe -DPYTHON3_INCLUDE_DIR=%envsDir%\OpenCV-%cvVersion%-py3\include -DPYTHON3_LIBRARY=%envsDir%\OpenCV-%cvVersion%-py3\libs\python36.lib -DPYTHON3_NUMPY_INCLUDE_DIRS=%envsDir%\OpenCV-%cvVersion%-py3\Lib\site-packages\numpy\core\include -DPYTHON3_PACKAGES_PATH=%envsDir%\OpenCV-%cvVersion%-py3\Lib\site-packages ..
::============================================::
::Compile OpenCV in release mode
cmake.exe --build . --config Release --target INSTALL
::xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx::
cd ..
cd ..
cd ..
