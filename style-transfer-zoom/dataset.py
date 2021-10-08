import torch
from torchvision import models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import config as cfg
from PIL import Image
import numpy as np
import os

class ImageNetData(Dataset):
	def __init__(self, image_paths, labels=None, size=[320, 240]):
		"""
		image_paths: a list of N paths for images in training set
		labels: soft targets for images as numpy array of shape (N, 1000)
		"""
		super(ImageNetData, self).__init__()
		self.image_paths=image_paths
		self.labels=labels
		self.inputsize=size
		self.transforms=self.random_transforms()
		if self.labels is not None:
			assert len(self.image_paths)==self.labels.shape[0]
			#number of images and soft targets should be the same

	def random_transforms(self):
		normalize_transform=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		#define normalization transform with which the torchvision models
		#were trained

		affine=T.RandomAffine(degrees=5, translate=(0.05, 0.05))
		hflip =T.RandomHorizontalFlip(p=0.7)
		#webcam output often has horizontal flips, we would like our network
		#to be resilient to horizontal flips
		blur=T.GaussianBlur(5) #kernel size 5x5

		rt1=T.Compose([T.Resize(self.inputsize), affine, T.ToTensor(), normalize_transform])
		rt2=T.Compose([T.Resize(self.inputsize), hflip, T.ToTensor(), normalize_transform])
		rt3=T.Compose([T.Resize(self.inputsize), blur, T.ToTensor(), normalize_transform])

		return [rt1, rt2, rt3]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		imgpath=self.image_paths[index]
		img=Image.open(imgpath).convert('RGB') 
		#some images are grayscale and need to be converted into RGB
	
		img_tensor=self.transforms[torch.randint(0,3,[1,1]).item()](img)

		if self.labels is None:
			return img_tensor
		else:
			label_tensor=torch.tensor(self.labels[index,:])
			return img_tensor, label_tensor

class DataManager(object):
	def __init__(self,imgpathfile, labelpath=None, size=[320, 240], use_test_data=False):
		"""
		imgpathfile: a text file containing paths of all images in the dataset
		stored as a list containting three lists for train, valid, test splits
		ex: [[p1,p2,p6...],[p3,p4...],[p5...]]

		labelpath (optional): path of .npy file which has a numpy array 
		of size (N, 1000) containing pre-computed soft targets
		The order of soft targets in the numpy array should correspond to
		the order of images in imgpathfile 

		size (2-list): [width, height] to which all images will be resized
		use_test_data (bool): whether or not to use test data (generally test data is used
		only once after you have verified model architecture and hyperparameters on validation dataset)
		"""

		self.imgpathfile=imgpathfile
		self.labelpath=labelpath
		self.imgsize=size

		assert os.path.exists(self.imgpathfile), 'File {} does not exist'.format(self.imgpathfile)

		self.dataloaders=self.get_data_loaders(use_test_data)

	def get_data_loaders(self, test=False):
		"""
		test (bool): whether or not to get test data loader
		"""

		with open(self.imgpathfile,'r') as f:
			train_paths, valid_paths, test_paths= eval(f.read())

		if self.labelpath is not None:
			all_labels=np.load(self.labelpath)

			assert all_labels.shape[0]== (len(train_paths)+len(valid_paths)+len(test_paths))

			train_labels=all_labels[:len(train_paths),:]
			valid_labels=all_labels[len(train_paths):len(train_paths)+len(valid_paths),:]
			test_labels=all_labels[-len(test_paths):,:]

		else:
			train_labels=None
			valid_labels=None
			test_labels=None

		train_data=ImageNetData(train_paths, train_labels, self.imgsize)
		valid_data=ImageNetData(valid_paths, valid_labels, self.imgsize)
		
		train_loader=DataLoader(train_data, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
		valid_loader=DataLoader(valid_data, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
		#evaluation of network (validation) does not require storing gradients, so GPU memory is freed up
		#therefore, validation can be performed at roughly twice the batch size of training for most
		#networks and GPUs. This reduces training time by doubling the throughput of validation

		if test:
			test_data=ImageNetData(test_paths, test_labels, self.imgsize)
			test_loader=DataLoader(test_data, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

			return train_loader, valid_loader, test_loader

		return train_loader, valid_loader
