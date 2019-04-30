
## Colorization

Run the getModels.sh file from command line to download the needed model files

```bash
sudo chmod a+x getModels.sh
./getModels.sh
```
  
### Python:

Commandline usage to colorize

a single image:
```bash
python3 colorizeImage.py --input greyscaleImage.png
```


a video file:
```bash
python3 colorizeVideo.py --input greyscaleVideo.mp4
```

  
### C++:

Compilation examples:

```bash
g++ -ggdb `pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.4.2/lib/pkgconfig/opencv.pc` colorizeImage.cpp -o colorizeImage.out
```

```bash
g++ -ggdb `pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.4.2/lib/pkgconfig/opencv.pc` colorizeVideo.cpp -o colorizeVideo.out
```
  
Commandline usage to colorize

a single image:
```bash
./colorizeImage.out greyscaleImage.png
```
a video file:
```bash
./colorizeVideo.out greyscaleVideo.mp4
```
