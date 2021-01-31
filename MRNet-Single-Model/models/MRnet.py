import torch
import torch.nn as nn
from torchvision import models
import os

class MRnet(nn.Module):
    """MRnet uses pretrained resnet50 as a backbone to extract features
    """
    
    def __init__(self):
        """This function will be used to initialize the 
        MRnet instance."""
        # Initialize nn.Module instance
        super(MRnet,self).__init__()

        # Initialize three backbones for three axis
        # All the three axes will use pretrained AlexNet model
        # The models will be used for extracting features from
        # the input images
        self.axial = models.alexnet(pretrained=True).features
        self.coronal = models.alexnet(pretrained=True).features
        self.saggital = models.alexnet(pretrained=True).features
        
        # Initialize 2D Adaptive Average Pooling layers
        # The pooling layers will reduce the size of
        # feature maps extracted from the previous axes
        self.pool_axial = nn.AdaptiveAvgPool2d(1)
        self.pool_coronal = nn.AdaptiveAvgPool2d(1)
        self.pool_saggital = nn.AdaptiveAvgPool2d(1)
        
        # Initialize a sequential neural network with
        # a single fully connected linear layer
        # The network will output the probability of
        # having a particular disease
        self.fc = nn.Sequential(
            nn.Linear(in_features=3*256,out_features=1)
        )

    def forward(self,x):
        """ Input is given in the form of `[image1, image2, image3]` where
        `image1 = [1, slices, 3, 224, 224]`. Note that `1` is due to the 
        dataloader assigning it a single batch. 
        """

        # squeeze the first dimension as there
        # is only one patient in each batch
        images = [torch.squeeze(img, dim=0) for img in x]
        
        # Extract features across each of the three plane
        # using the three pre-trained AlexNet models defined earlier
        image1 = self.axial(images[0])
        image2 = self.coronal(images[1])
        image3 = self.saggital(images[2])

        # Convert the image dimesnsions from [slices, 256, 1, 1] to
        # [slices,256]
        image1 = self.pool_axial(image1).view(image1.size(0), -1)
        image2 = self.pool_coronal(image2).view(image2.size(0), -1)
        image3 = self.pool_saggital(image3).view(image3.size(0), -1)

        # Find maximum value in each slice
        # This will reduce the dimensions of image to [1,256]
        # This is done in order to keep only the most prevalent
        # features for each slice
        image1 = torch.max(image1,dim=0,keepdim=True)[0]
        image2 = torch.max(image2,dim=0,keepdim=True)[0]
        image3 = torch.max(image3,dim=0,keepdim=True)[0]

        # Stack the 3 images together to create the output
        # of size [1, 256*3]
        output = torch.cat([image1,image2,image3], dim=1)

        # Feed the output to the sequential network created earlier
        # The network will return a probability of having a specific
        # disease
        output = self.fc(output)
        return output

    def _load_wieghts(self):
        """load pretrained weights"""
        pass
