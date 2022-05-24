import os
from argparse import ArgumentParser

import tensorflow as tf

from tools import get_images_paths


def _byte_feature(value):
    """Convert string / byte into bytes_list."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList can't unpack string from EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Convert bool / enum / int / uint into int64_list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label):
    feature = {
        "label": _int64_feature(label),
        "image_raw": _byte_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def store_many_tfrecords(images_list, save_file):

    assert save_file.endswith(
        ".tfrecords",
    ), 'File path is wrong, it should contain "*myname*.tfrecords"'

    directory = os.path.dirname(save_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with tf.io.TFRecordWriter(save_file) as writer:  # start writer
        for label, filename in enumerate(images_list):  # cycle by each image path
            image_string = open(filename, "rb").read()  # read the image as bytes string
            tf_example = image_example(
                image_string, label,
            )  # save the data as tf.Example object
            writer.write(tf_example.SerializeToString())  # and write it into database


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="path to the images folder to collect",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help='path to the output tfrecords file i.e. "path/to/folder/myname.tfrecords"',
    )

    args = parser.parse_args()
    image_paths = get_images_paths(args.path)
    store_many_tfrecords(image_paths, args.output)
