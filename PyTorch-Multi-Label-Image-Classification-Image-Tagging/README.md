# Setup
Before installation create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
Install the dependencies
```bash
pip install -r requirements.txt
```

# Training
For training run [jupyter notebook](Pipeline.ipynb)

# Additional instructions
## Data preparation
We use [the NUS-WIDE dataset](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) for this tutorial. 
Instead of parsing Flickr for image downloading we use [a dump](https://drive.google.com/open?id=0B7IzDz-4yH_HSmpjSTlFeUlSS00) from [this github repository](https://github.com/thuml/HashNet/tree/master/pytorch#datasets)
Download and extract it.

Also, we added pre-processed annotations:  
```nus_wide/train.json```  
```nus_wide/test.json``` 

If you want to create them yourself, run the command:
```bash
python split_data_nus.py -i images
```
where ``` -i images``` is the path to the folder with extracted images

## Subset creation
You can train the model for the entire data set, but it takes a lot of time. For this tutorial we use part of this data.

For subset creation run the command: 
```bash
python create_subset.py -i images
```
where ``` -i images``` is the path to the folder with extracted images

Additional options:
```bash
python create_subset.py -h
usage: Subset creation [-h] -i IMG_PATH [-v VAL_SIZE] [-t TRAIN_SIZE]
                       [--shuffle] [-l LABELS [LABELS ...]]

optional arguments:
  -h, --help            show this help message and exit
  -i IMG_PATH, --img-path IMG_PATH
                        Path to the "images" folder
  -v VAL_SIZE, --val-size VAL_SIZE
                        Size of the validation data
  -t TRAIN_SIZE, --train-size TRAIN_SIZE
                        Size of the train data
  --shuffle             Shuffle samples before splitting
  -l LABELS [LABELS ...], --labels LABELS [LABELS ...]
                        Subset labels
```
