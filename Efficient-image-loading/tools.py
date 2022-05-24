import os


def get_images_paths(d):
    return [os.path.join(d, f) for f in os.listdir(d)]
