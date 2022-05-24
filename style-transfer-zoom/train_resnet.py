import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch
import numpy as np
from dataset import DataManager
import config as cfg
import time
from torch.utils.tensorboard import SummaryWriter
from apex import amp

class SoftTargetLoss(nn.Module):
	def __init__(self, temperature=1):
		"""
		Soft Target Loss as introduced by Hinton et. al. 
		in https://arxiv.org/abs/1503.02531
		temp (float or int): annealing temperature hyperparameter
		temperature=1 corresponds to usual softmax
		"""
		super(SoftTargetLoss, self).__init__()
		self.register_buffer('temperature', torch.tensor(temperature))
		#temperature

	def forward(self, student_logits, teacher_logits):
		student_probabilities=nn.functional.softmax(student_logits/self.temperature)
		teacher_probabilities=nn.functional.softmax(teacher_logits/self.temperature)

		loss = - torch.mul(teacher_probabilities, torch.log(student_probabilities))

		return torch.mean(loss)

class Trainer(object):
	def __init__(self, net, manager, savepath):
		"""
		net(nn.Module): Neural network to be trained
		
		manager(DataManager): data manager from dataset.py
		
		savepath(str): a format-ready string like 'model_{}.path'
		for which .format method can be called while saving models
		at every epoch
		"""

		self.net=net
		self.manager=manager
		self.savepath=savepath #should have curly brackets, ex. 'model_{}.pth'

		self.criterion = SoftTargetLoss(cfg.TEMPERATURE)
		self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LR)
		self.writer=SummaryWriter()

	def save(self, path):
		checkpoint= {'model':self.net.state_dict(), 
		'optimizer':self.optimizer.state_dict(),
		'amp':amp.state_dict() }

		torch.save(checkpoint, path)
		print(f'Saved model to {path}')

	def train(self, epochs=None, evaluate_interval=None):
		steps=0

		epochs=epochs if epochs else cfg.EPOCHS
		evaluate_interval=evaluate_interval if evaluate_interval else cfg.EVAL_INTERVAL

		device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		
		if device.type != 'cuda':
			print(f'GPU not found. Training will be done on device of type {device.type}')

		self.net.to(device)

		self.net, self.optimizer = amp.initialize(self.net, self.optimizer,
									opt_level='O2')

		self.net.train()

		train_iterator, valid_iterator, *_ = self.manager.dataloaders

		get_top5_accuracy=lambda p,y: (torch.topk(p, 5, dim=1).indices == torch.argmax(y, 1)[:,None]).sum(dim=1).to(torch.float).mean().item()
		mean= lambda v: sum(v)/len(v)

		for epoch in range(epochs):
			start_time=time.time()

			for idx, (x,y) in enumerate(train_iterator):
				self.optimizer.zero_grad()
				#print('Resnet input shape= ', x.shape)
				x=x.to(device)
				y=y.to(device)

				preds=self.net(x)

				loss=self.criterion(preds, y)

				#loss.backward()
				with amp.scale_loss(loss, self.optimizer) as scaled_loss:
					scaled_loss.backward()

				self.optimizer.step()

				top5_accuracy=get_top5_accuracy(preds, y)
				#this isn't *really* the top 5 accuracy because it is evaluated against the outputs of the teacher
				#model as opposed to ground truth labels. Since the value of the loss is not easy to grasp 
				#intuitively, this proxy serves as an easily computable metric to monitor the progress of the 
				#student network, especially if the training data is also imagenet.

				self.writer.add_scalar('Loss', loss, steps)
				self.writer.add_scalar('Top-5 training accuracy', top5_accuracy, steps)

				steps+=1

				if steps%evaluate_interval==0:
					valid_loss=[]
					valid_accuracy=[]
					self.net.eval() #put network in evaluation mode

					with torch.no_grad():
						for xv, yv in valid_iterator:
							xv=xv.to(device)
							yv=yv.to(device)
							preds=self.net(xv)
							vtop5a=get_top5_accuracy(preds, yv)

							vloss=self.criterion(preds, yv)

							valid_loss.append(vloss.item())
							valid_accuracy.append(vtop5a)

					self.writer.add_scalar('Validation Loss', mean(valid_loss), steps)
					self.writer.add_scalar('Top-5 validation accuracy', mean(valid_accuracy), steps)
					self.writer.flush()

					self.net.train() #return to training mode
					pass

			self.writer.flush() #make sure the writer updates all stats until now
			self.save(self.savepath.format(epoch))
			end_time=time.time()
			print('Time taken for last epoch = {:.3f} seconds'.format(end_time-start_time))


def main():
	manager=DataManager(cfg.IMGPATH_FILE, cfg.SOFT_TARGET_PATH, cfg.SIZE)
	net=models.resnet18(pretrained=False)
	trainer=Trainer(net, manager, cfg.SAVE_PATH)
	trainer.train()

if __name__=="__main__":
	main()
