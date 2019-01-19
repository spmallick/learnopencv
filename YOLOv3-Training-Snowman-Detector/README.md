Training YOLOv3 Object Detector - Snowman

1. Install awscli

`sudo pip3 install awscli` 

2. Download the images from OpenImagesV4

`python3 getDataFromOpenImages_snowman.py`

3. Create the train-test split

`python3 splitTrainAndTest.py /data-ssd/sunita/snowman/JPEGImages`

Give the correct path to the data JPEGImages folder. The 'labels' folder should be in the same directory as the JPEGImages folder.

4. Install Darknet and compile it.
`
cd ~
git clone https://github.com/pjreddie/darknet
cd darknet
make
`
5. Get the pretrained model

`wget https://pjreddie.com/media/files/darknet53.conv.74 -O ~/darknet/darknet53.conv.74`

6. Fill in correct paths in the darknet.data file

7. Start the training as below, by giving the correct paths to all the files being used as arguments

`cd ~/darknet`

`./darknet detector train /path/to/snowman/darknet.data  /path/to/snowman/darknet-yolov3.cfg ./darknet53.conv.74 > /path/to/snowman/train.log`

8. Give the correct path to the modelConfiguration and modelWeights files in object_detection_yolo.py and test any image or video for snowman detection, e.g.

`python3 object_detection_yolo.py --image=snowmanImage.jpg`

