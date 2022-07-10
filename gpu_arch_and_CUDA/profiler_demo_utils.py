import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

try:
	from apex import amp
	has_apex=True
except:
	print('apex not available')
	has_apex=False

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

## THIS IS THE IMPORTANT BIT
from torch.profiler import profile, record_function, ProfilerActivity, schedule


class CIFAR10_Manager(object):
	def __init__(self, indir, bsize=128):
		self.indir=indir
		self.inputsize=(32,32)
		self.input_transforms=self.get_input_transforms()
		self.batchsize=bsize
		self.train_loader=self.get_train_loader()
		self.valid_loader=self.get_valid_loader()
		
	def get_train_loader(self):
		pass
		tdata=datasets.CIFAR10(
			root=self.indir, 
			train=True, 
			transform=self.input_transforms,
			download=True)

		tloader=DataLoader(tdata, self.batchsize, shuffle=True, num_workers=8)
		return tloader
		
	def get_valid_loader(self):
		pass
		vdata=datasets.CIFAR10(
			root=self.indir, 
			train=False, 
			transform=self.input_transforms,
			download=True)

		vloader=DataLoader(vdata, self.batchsize, shuffle=True, num_workers=8)
		return vloader

	def get_input_transforms(self):
		normalize_transform=T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

		affine=T.RandomAffine(degrees=5, translate=(0.05, 0.05))
		hflip =T.RandomHorizontalFlip(p=0.7)
		vflip =T.RandomVerticalFlip(p=0.7)
		
		blur=T.GaussianBlur(5) #kernel size 5x5

		composed=T.Compose([T.Resize(self.inputsize), affine,hflip, vflip, blur, T.ToTensor(), normalize_transform])

		return composed
		
class VisionClassifier(nn.Module):
	def __init__(self, nclasses, mname='resnet18'):
		super(VisionClassifier, self).__init__()
		self.nclasses=nclasses
		mdict={
		'resnet18':models.resnet18, 
		'resnet50':models.resnet50, 
		'mobilenetv3':models.mobilenet_v3_large,
		'densenet':models.densenet121,
		'squeezenet':models.squeezenet1_0,
		'inception':models.inception_v3,
		}

		mhandle=mdict.get(mname, None)
		if not mhandle:
			print(f'Model {mname} not supported. Supportd models are: {mdict.keys()}')
			quit()
		else:
			print(f'Initializing {mname}')

		fullmodel=mhandle(pretrained=True)

		self.backbone=nn.Sequential(*list(fullmodel.children())[:-1])
		self.flatten=nn.Flatten()
		hidden_dim=list(fullmodel.children())[-1].in_features
		self.linear=nn.Linear(hidden_dim, self.nclasses)

	def forward(self, x):
		x=self.backbone(x)
		x=self.flatten(x)
		x=self.linear(x)
		return x