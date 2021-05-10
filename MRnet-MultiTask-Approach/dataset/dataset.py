import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data

from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class MRData():
    """This class used to load MRnet dataset from `./images` dir
    """

    def __init__(self,task = 'acl', train = True, transform = None, weights = None):
        """Initialize the dataset

        Args:
            plane : along which plane to load the data
            task : for which task to load the labels
            train : whether to load the train or val data
            transform : which transforms to apply
            weights (Tensor) : Give wieghted loss to postive class eg. `weights=torch.tensor([2.223])`
        """
        self.planes=['axial', 'coronal', 'sagittal']
        self.diseases = ['abnormal','acl','meniscus']
        self.records = {'abnormal' : None, 'acl' : None, 'meniscus' : None}
        # an empty dictionary
        self.image_path={}
        
        if train:
            for disease in self.diseases:
                self.records[disease] = pd.read_csv('./images/train-{}.csv'.format(disease),header=None, names=['id', 'label'])

            '''
            self.image_path[<plane>]= dictionary {<plane>: path to folder containing
                                                                image for that plane}
            '''
            for plane in self.planes:
                self.image_path[plane] = './images/train/{}/'.format(plane)
        else:
            for disease in self.diseases:
                self.records[disease] = pd.read_csv('./images/valid-{}.csv'.format(disease),header=None, names=['id', 'label'])

            '''
            self.image_path[<plane>]= dictionary {<plane>: path to folder containing
                                                                image for that plane}
            '''
            for plane in self.planes:
                self.image_path[plane] = './images/valid/{}/'.format(plane)

        
        self.transform = transform 

        for disease in self.diseases:
            self.records[disease]['id'] = self.records[disease]['id'].map(
                lambda i: '0' * (4 - len(str(i))) + str(i))
        
        # empty dictionary
        self.paths={}    
        for plane in self.planes:
            self.paths[plane] = [self.image_path[plane] + filename +
                        '.npy' for filename in self.records['acl']['id'].tolist()]

        self.labels = {'abnormal' : None, 'acl' : None, 'meniscus' : None}
        for disease in self.diseases:
            self.labels[disease] = self.records[disease]['label'].tolist()

        weights_ = []
        for disease in self.diseases:
            pos = sum(self.labels[disease])
            neg = len(self.labels[disease]) - pos
            weights_.append(neg/pos)

        # Find the wieghts of pos and neg classes
        if weights:
            self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.FloatTensor(weights_)
        
        print('Weights for loss is : ', self.weights)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.records['acl'])

    def __getitem__(self, index):
        """
        Returns `(images,labels)` pair
        where image is a list [imgsPlane1,imgsPlane2,imgsPlane3]
        and labels is a list [gt,gt,gt]
        """
        img_raw = {}
        
        for plane in self.planes:
            img_raw[plane] = np.load(self.paths[plane][index])
            img_raw[plane] = self._resize_image(img_raw[plane])
            
        label = []
        for disease in self.diseases:
            label.append(self.labels[disease][index])

        label = torch.FloatTensor(label)

        return [img_raw[plane] for plane in self.planes], label

    def _resize_image(self, image):
        """Resize the image to `(3,224,224)` and apply 
        transforms if possible.
        """
        # Resize the image
        pad = int((image.shape[2] - INPUT_DIM)/2)
        image = image[:,pad:-pad,pad:-pad]
        image = (image-np.min(image))/(np.max(image)-np.min(image))*MAX_PIXEL_VAL
        image = (image - MEAN) / STDDEV

        if self.transform:
            image = self.transform(image)
        else:
            image = np.stack((image,)*3, axis=1)
        
        image = torch.FloatTensor(image)
        return image

def load_data(task : str):

    # Define the Augmentation here only
    augments = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        RandomFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])

    print('Loading Train Dataset of {} task...'.format(task))
    train_data = MRData(task, train=True, transform=augments)
    train_loader = data.DataLoader(
        train_data, batch_size=1, num_workers=4, shuffle=True
    )

    print('Loading Validation Dataset of {} task...'.format(task))
    val_data = MRData(task, train=False)
    val_loader = data.DataLoader(
        val_data, batch_size=1, num_workers=4, shuffle=False
    )

    return train_loader, val_loader, train_data.weights, val_data.weights
