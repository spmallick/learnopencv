import os
from abc import abstractmethod
from timeit import default_timer as timer

import cv2
import lmdb
import numpy as np
import tensorflow as tf
from PIL import Image
from turbojpeg import TurboJPEG

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ImageLoader:
    extensions: tuple = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".tfrecords")

    def __init__(self, path: str, mode: str = "BGR"):
        self.path = path
        self.mode = mode
        self.dataset = self.parse_input(self.path)
        self.sample_idx = 0

    def parse_input(self, path):

        # single image or tfrecords file
        if os.path.isfile(path):
            assert path.lower().endswith(
                self.extensions,
            ), f"Unsupportable extension, please, use one of {self.extensions}"
            return [path]

        if os.path.isdir(path):
            # lmdb environment
            if any([file.endswith(".mdb") for file in os.listdir(path)]):
                return path
            else:
                # folder with images
                paths = [os.path.join(path, image) for image in os.listdir(path)]
                return paths

    def __iter__(self):
        self.sample_idx = 0
        return self

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def __next__(self):
        pass


class CV2Loader(ImageLoader):
    def __next__(self):
        start = timer()
        path = self.dataset[self.sample_idx]  # get image path by index from the dataset
        image = cv2.imread(path)  # read the image
        full_time = timer() - start
        if self.mode == "RGB":
            start = timer()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change color mode
            full_time += timer() - start
        self.sample_idx += 1
        return image, full_time


class PILLoader(ImageLoader):
    def __next__(self):
        start = timer()
        path = self.dataset[self.sample_idx]  # get image path by index from the dataset
        image = np.asarray(Image.open(path))  # read the image as numpy array
        full_time = timer() - start
        if self.mode == "BGR":
            start = timer()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # change color mode
            full_time += timer() - start
        self.sample_idx += 1
        return image, full_time


class TurboJpegLoader(ImageLoader):
    def __init__(self, path, **kwargs):
        super(TurboJpegLoader, self).__init__(path, **kwargs)
        self.jpeg_reader = TurboJPEG()  # create TurboJPEG object for image reading

    def __next__(self):
        start = timer()
        file = open(self.dataset[self.sample_idx], "rb")  # open the input file as bytes
        full_time = timer() - start
        if self.mode == "RGB":
            mode = 0
        elif self.mode == "BGR":
            mode = 1
        start = timer()
        image = self.jpeg_reader.decode(file.read(), mode)  # decode raw image
        full_time += timer() - start
        self.sample_idx += 1
        return image, full_time


class LmdbLoader(ImageLoader):
    def __init__(self, path, **kwargs):
        super(LmdbLoader, self).__init__(path, **kwargs)
        self.path = path
        self._dataset_size = 0
        self.dataset = self.open_database()

    # we need to open the database to read images from it
    def open_database(self):
        lmdb_env = lmdb.open(self.path)  # open the environment by path
        lmdb_txn = lmdb_env.begin()  # start reading
        lmdb_cursor = lmdb_txn.cursor()  # create cursor to iterate through the database
        self._dataset_size = lmdb_env.stat()[
            "entries"
        ]  # get number of items in full dataset
        return lmdb_cursor

    def __iter__(self):
        self.dataset.first()  # return the cursor to the first database element
        return self

    def __next__(self):
        start = timer()
        raw_image = self.dataset.value()  # get raw image
        image = np.frombuffer(raw_image, dtype=np.uint8)  # convert it to numpy
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # decode image
        full_time = timer() - start
        if self.mode == "RGB":
            start = timer()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            full_time += timer() - start
        start = timer()
        self.dataset.next()  # step to the next element in database
        full_time += timer() - start
        return image, full_time

    def __len__(self):
        return self._dataset_size  # get dataset length


class TFRecordsLoader(ImageLoader):
    def __init__(self, path, **kwargs):
        super(TFRecordsLoader, self).__init__(path, **kwargs)
        self._dataset = self.open_database()

    def open_database(self):
        def _parse_image_function(example_proto):
            return tf.io.parse_single_example(example_proto, image_feature_description)

        # dataset structure description
        image_feature_description = {
            "label": tf.io.FixedLenFeature([], tf.int64),
            "image_raw": tf.io.FixedLenFeature([], tf.string),
        }
        raw_image_dataset = tf.data.TFRecordDataset(self.path)  # open dataset by path
        parsed_image_dataset = raw_image_dataset.map(
            _parse_image_function,
        )  # parse dataset using structure description

        return parsed_image_dataset

    def __iter__(self):
        self.dataset = self._dataset.as_numpy_iterator()
        return self

    def __next__(self):
        start = timer()
        value = next(self.dataset)[
            "image_raw"
        ]  # step to the next element in database and get new image
        image = tf.image.decode_jpeg(value).numpy()  # decode raw image
        full_time = timer() - start
        if self.mode == "BGR":
            start = timer()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            full_time += timer() - start
        return image, full_time

    def __len__(self):
        return self._dataset.reduce(
            np.int64(0), lambda x, _: x + 1,
        ).numpy()  # get dataset length


methods = {
    "cv2": CV2Loader,
    "pil": PILLoader,
    "turbojpeg": TurboJpegLoader,
    "lmdb": LmdbLoader,
    "tfrecords": TFRecordsLoader,
}
