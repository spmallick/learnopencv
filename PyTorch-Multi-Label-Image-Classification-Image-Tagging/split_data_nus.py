import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser('Data preparation script')
    parser.add_argument("-i", "--img-path", required=True, type=str, help='Path to "image" folder')
    args = parser.parse_args()
    img_path = args.img_path
    with open('nus_wide/cats') as f:
        classes = np.array(f.read().split('\n'))
    with open('nus_wide/database.txt') as f:
        all_samples_data = f.read().splitlines()
    with open('nus_wide/NUS-WIDE-urls.txt') as f:
        urls = f.read().splitlines()
    with open('nus_wide/TestImagelist.txt') as f:
        test_images = f.read().splitlines()

    test_images = [item.split('\\')[-1] for item in test_images]
    name_map = np.array([[item.split()[0].split('\\')[-1],
                          item.split()[4].split('/')[-1]] for item in urls[1:]])
    test_samples = []
    train_samples = []
    for data_line in tqdm(all_samples_data):
        src_img = os.path.basename(data_line.split(' ')[0])
        item_labels = np.array(data_line.split(' ')[1:], dtype=int)
        image_name_raw = os.path.join(img_path, src_img)

        try:
            Image.open(image_name_raw)
        except:
            continue

        if np.sum(item_labels) == 0:
            continue

        is_test_item = False
        index = [index for index, item in enumerate(name_map[:, 1]) if item in src_img]
        if len(index) == 1:
            orig_name = name_map[index[0]][0]
            is_test_item = orig_name in test_images

        item = {'image_name': image_name_raw, 'image_labels': classes[np.argwhere(item_labels > 0)[:, 0]].tolist()}
        if is_test_item:
            test_samples.append(item)
        else:
            train_samples.append(item)

    print('Test size:', len(test_samples), 'Train size:', len(train_samples))

    with open(os.path.join(img_path, 'train.json'), 'w') as fp:
        json.dump({'samples': train_samples, 'labels': classes}, fp, indent=3)

    with open(os.path.join(img_path, 'test.json'), 'w') as fp:
        json.dump({'samples': test_samples, 'labels': classes}, fp, indent=3)


if __name__ == '__main__':
    main()
