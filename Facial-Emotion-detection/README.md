# Facial-Emotion-Detection

 <img src = "https://github.com/gulshan-mittal/Facial-Emotion-Detection/blob/master/emotions1.png?raw=true" width= 20% height =15% align='right'>

### Set up the environment

* ``virtualvenv -p /usr/bin/python3.5+ emoji``
* ``source emoji/bin/activate``
* ``pip3 install -r requirements.txt``

### How to Run

* For visualisation or testing on the pre-trained model
  * ``python3.5+ demo_emotion_detection.py --image ./images/ --result ./results/``
* For training the model on Fer2013 dataset
  *  ``python3 train_emotion_detection.py --epochs 100 --batch_size 64``
  * Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
  * Move the downloaded file to the data sets directory inside this repository.
  * Untar the file:
    * ``tar -xzf fer2013.tar``
    * Run the above command ``python3 train_emotion_detection.py --epochs 100 --batch_size 64``

For more details about **Network Architecture & Model** refer [Mini_Xception, 2017](https://arxiv.org/pdf/1710.07557.pdf)

### Testing the Model on Images

<img src = "https://github.com/gulshan-mittal/Facial-Emotion-Detection/blob/master/results/result_test_3.png?raw=true">
