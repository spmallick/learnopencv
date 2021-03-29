# Setup

This code was tested with python 3.7, however, it should work with any python 3.

1. Create and activate virtual environment for experiments with t-SNE.

```bash
python3 -m venv venv
source venv/bin/activate
```

2. install the dependencies

```bash
python3 -m pip install -r requirements.txt
```

# Data downloading

Download data from Kaggle and unzip it.
The easiest way is to use kaggle console API. To setup it, follow [this guide](https://www.kaggle.com/general/74235).
However, you can download the data using your browser - results will be the same.

After that, execute the following commands:

```bash

kaggle datasets download alessiocorrado99/animals10

mkdir -p data

cd data

unzip ../animals10.zip

cd ..

```

# Executing the T-SNE visualization

```bash

python3 tsne.py

```

Additional options:

```bash
python3 tsne.py -h

usage: tsne.py [-h] [--path PATH] [--batch BATCH] [--num_images NUM_IMAGES]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH
  --batch BATCH
  --num_images NUM_IMAGES

```

You can change the data directory with `--path` argument.

Tweak the `--num_images` to speed-up the process - by default it is 500, you can make it smaller.

Tweak the `--batch` to better utilize your PC's resources. The script uses GPU automatically if it available. You may
want to increase the batch size to utilize the GPU better or decrease it if the default batch size does not fit your
GPU.
