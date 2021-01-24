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
        # Define the three planes to use
        self.planes=['axial', 'coronal', 'sagittal']
        # Initialize the records as None
        self.records = None
        # an empty dictionary
        self.image_path={}
        
        # If we are in training loop
        if train:
            # Read data about patient records
            self.records = pd.read_csv('./images/train-{}.csv'.format(task),header=None, names=['id', 'label'])

            for plane in self.planes:
                # For each plane, specify the image path
                self.image_path[plane] = './images/train/{}/'.format(plane)
        else:
            # If we are in testing loop
            # don't use any transformation
            transform = None
            # Read testing/validation data (patients records)
            self.records = pd.read_csv('./images/valid-{}.csv'.format(task),header=None, names=['id', 'label'])
            
            for plane in self.planes:
                # Read path of images for each plane
                self.image_path[plane] = './images/valid/{}/'.format(plane)

        # Initialize the transformation to apply on images
        self.transform = transform 

        # Append 0s to the patient record id
        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        # empty dictionary
        self.paths={}
        for plane in self.planes:
            # Get paths of numpy data files for each plane
            self.paths[plane] = [self.image_path[plane] + filename +
                          '.npy' for filename in self.records['id'].tolist()]

        # Convert labels from Pandas Series to a list
        self.labels = self.records['label'].tolist()

        # Total positive cases
        pos = sum(self.labels)
        # Total negative cases
        neg = len(self.labels) - pos

        # Find the wieghts of pos and neg classes
        if weights:
            self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.FloatTensor([neg / pos])
        
        print('Number of -ve samples : ', neg)
        print('Number of +ve samples : ', pos)
        print('Weights for loss is : ', self.weights)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.records)

    def __getitem__(self, index):
        """
        Returns `(images,labels)` pair
        where image is a list [imgsPlane1,imgsPlane2,imgsPlane3]
        and labels is a list [gt,gt,gt]
        """
        img_raw = {}
        
        for plane in self.planes:
            # Load raw image data for each plane
            img_raw[plane] = np.load(self.paths[plane][index])
            # Resize the image loaded in the previous step
            img_raw[plane] = self._resize_image(img_raw[plane])
            
        label = self.labels[index]
        # Convert label to 0 and 1
        if label == 1:
            label = torch.FloatTensor([1])
        elif label == 0:
            label = torch.FloatTensor([0])

        # Return a list of three images for three planes and the label of the record
        return [img_raw[plane] for plane in self.planes], label

    def _resize_image(self, image):
        """Resize the image to `(3,224,224)` and apply 
        transforms if possible.
        """
        # Resize the image
        # Calculate extra padding present in the image
        # which needs to be removed
        pad = int((image.shape[2] - INPUT_DIM)/2)
        # This is equivalent to center cropping the image
        image = image[:,pad:-pad,pad:-pad]
        # Normalize the image by subtracting it by mean and dividing by standard
        # deviation
        image = (image-np.min(image))/(np.max(image)-np.min(image))*MAX_PIXEL_VAL
        image = (image - MEAN) / STDDEV
        
        # If the transformation is not None
        if self.transform:
            # Transform the image based on the specified transformation
            image = self.transform(image)
        else:
            # Else, just stack the image with itself in order to match the required
            # dimensions
            image = np.stack((image,)*3, axis=1)
        # Convert the image to a FloatTensor and return it
        image = torch.FloatTensor(image)
        return image

def load_data(task : str):

    # Define the Augmentation here only
    augments = Compose([
        # Convert the image to Tensor
        transforms.Lambda(lambda x: torch.Tensor(x)),
        # Randomly rotate the image with an angle
        # between -25 degrees to 25 degrees
        RandomRotate(25),
        # Randomly translate the image by 11% of 
        # image height and width
        RandomTranslate([0.11, 0.11]),
        # Randomly flip the image
        RandomFlip(),
        # Change the order of image channels
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])

    print('Loading Train Dataset of {} task...'.format(task))
    # Load training dataset
    train_data = MRData(task, train=True, transform=augments)
    train_loader = data.DataLoader(
        train_data, batch_size=1, num_workers=11, shuffle=True
    )

    print('Loading Validation Dataset of {} task...'.format(task))
    # Load validation dataset
    val_data = MRData(task, train=False)
    val_loader = data.DataLoader(
        val_data, batch_size=1, num_workers=11, shuffle=False
    )

    return train_loader, val_loader, train_data.weights, val_data.weights
