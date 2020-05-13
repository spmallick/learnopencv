import argparse
import json
import os
from random import shuffle


def main():
    parser = argparse.ArgumentParser('Subset creation')
    parser.add_argument("-i", "--img-path", required=True, type=str, help='Path to the "images" folder')
    parser.add_argument("-v", "--val-size", default=1000, type=int, help='Size of the validation data')
    parser.add_argument("-t", "--train-size", default=5000, type=int, help='Size of the train data')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle samples before splitting')
    parser.add_argument("-l", "--labels", nargs='+', default=['house', 'birds', 'sun', 'valley',
                                                              'nighttime', 'boats', 'mountain', 'tree', 'snow',
                                                              'beach', 'vehicle', 'rocks',
                                                              'reflection', 'sunset', 'road', 'flowers', 'ocean',
                                                              'lake', 'window', 'plants',
                                                              'buildings', 'grass', 'water', 'animal', 'person',
                                                              'clouds', 'sky'], help='Subset labels')
    args = parser.parse_args()
    img_path = args.img_path
    labels = args.labels

    with open('nus_wide/cats') as l_f:
        possible_labels = l_f.readlines()
        possible_labels = [i.strip() for i in possible_labels]

    for label in labels:
        if label not in possible_labels:
            print('Label:', label, "is unknown. Possible labels:", ', '.join(possible_labels))
            exit(-1)

    with open(os.path.join(img_path, 'test.json')) as fp:
        test_data = json.load(fp)
    test_samples = test_data['samples']

    with open(os.path.join(img_path, 'train.json')) as fp:
        train_data = json.load(fp)
    train_samples = train_data['samples']

    if args.shuffle:
        shuffle(test_samples)
        shuffle(train_samples)

    train_size = args.train_size
    test_size = args.val_size

    small_train = []
    i = 0
    while len(small_train) < train_size:
        sample_img_path, sample_labels = train_samples[i]['image_name'], train_samples[i]['image_labels']
        sample_labels = [label for label in sample_labels if label in labels]
        if len(sample_labels):
            small_train.append({'image_name': sample_img_path, 'image_labels': sample_labels})
        i += 1

    small_test = []
    i = 0
    while len(small_test) < test_size:
        sample_img_path, sample_labels = test_samples[i]['image_name'], test_samples[i]['image_labels']
        sample_labels = [label for label in sample_labels if label in labels]
        if len(sample_labels):
            small_test.append({'image_name': sample_img_path, 'image_labels': sample_labels})
        i += 1

    with open(os.path.join(img_path, 'small_train.json'), 'w') as fp:
        json.dump({'samples': small_train, 'labels': labels}, fp, indent=3)

    with open(os.path.join(img_path, 'small_test.json'), 'w') as fp:
        json.dump({'samples': small_test, 'labels': labels}, fp, indent=3)


if __name__ == '__main__':
    main()
