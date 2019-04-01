### Download the Models

Run the **getModels.sh** file from command line to download the needed model files

	sudo chmod a+x getModels.sh
	./getModels.sh



### How to run the code

Command line usage for object detection using YOLOv3 

* Python

  * A single image:
    	

    ```bash
    python3 object_detection_yolo.py --image=bird.jpg
    ```

  * A video file:

       ```bash
       python3 object_detection_yolo.py --video=run.mp4
       ```

       

* C++:

  * A single image:
        

    ```bash
    ./object_detection_yolo.out --image=bird.jpg
    ```

    

  * A video file:

    ```bash
     ./object_detection_yolo.out --video=run.mp4
    ```



### Compilation examples

```bash
g++ -ggdb pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.4.2/lib/pkgconfig/opencv.pc object_detection_yolo.cpp -o object_detection_yolo.out
```



### Results of YOLOv3
<img src = "https://github.com/gulshan-mittal/learnopencv/blob/dev1/ObjectDetection-YOLO/bird_yolo_out_py.jpg" width = 400 height = 300/>
