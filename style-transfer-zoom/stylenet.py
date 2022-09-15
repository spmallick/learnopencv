import torch
import torch.nn as nn
from torch import optim
import time
import os
from PIL import Image
from torchvision import transforms as T
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from dataset import DataManager
import config as cfg
import sys

if not sys.platform=='darwin':
	from apex import amp

class StyleNetwork(nn.Module):
	def __init__(self, loadpath=None):
		super(StyleNetwork, self).__init__()
		self.loadpath=loadpath

		self.layer1 = self.get_conv_module(inc=3, outc=16, ksize=9)

		self.layer2 = self.get_conv_module(inc=16, outc=32)

		self.layer3 = self.get_conv_module(inc=32, outc=64)

		self.layer4 = self.get_conv_module(inc=64, outc=128)

		self.connector1=self.get_depthwise_separable_module(128, 128)

		self.connector2=self.get_depthwise_separable_module(64, 64)

		self.connector3=self.get_depthwise_separable_module(32, 32)

		self.layer5 = self.get_deconv_module(256, 64)

		self.layer6 = self.get_deconv_module(128, 32)

		self.layer7 = self.get_deconv_module(64, 16)

		self.layer8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

		self.activation=nn.Sigmoid()

		if self.loadpath:
			self.load_state_dict(torch.load(self.loadpath, map_location=torch.device('cpu')))

	def get_conv_module(self, inc, outc, ksize=3):
		padding=(ksize-1)//2
		conv=nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksize, stride=2, padding=padding)
		bn=nn.BatchNorm2d(outc)
		relu=nn.LeakyReLU(0.1)

		return nn.Sequential(conv, bn, relu)

	def get_deconv_module(self, inc, outc, ksize=3):
		padding=(ksize-1)//2
		tconv=nn.ConvTranspose2d(inc, outc, kernel_size=ksize, stride=2, padding=padding, output_padding=padding)
		bn=nn.BatchNorm2d(outc)
		relu=nn.LeakyReLU(0.1)

		return nn.Sequential(tconv, bn, relu)


	def get_depthwise_separable_module(self, inc, outc):
		"""
		inc(int): number of input channels
		outc(int): number of output channels

		Implements a depthwise separable convolution layer
		along with batch norm and activation.
		Intended to be used with inc=outc in the current architecture
		"""
		depthwise=nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, groups=inc)
		pointwise=nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, groups=1)
		bn_layer=nn.BatchNorm2d(outc)
		activation=nn.LeakyReLU(0.1)

		return nn.Sequential(depthwise, pointwise, bn_layer, activation)

	def forward(self, x):

		x=self.layer1(x)

		x2=self.layer2(x)

		x3=self.layer3(x2)

		x4=self.layer4(x3)

		xs4=self.connector1(x4)
		xs3=self.connector2(x3)
		xs2=self.connector3(x2)

		c1=torch.cat([x4, xs4], dim=1)

		x5=self.layer5(c1)

		c2=torch.cat([x5, xs3], dim=1)

		x6=self.layer6(c2)

		c3=torch.cat([x6, xs2], dim=1)

		x7=self.layer7(c3)

		out=self.layer8(x7)

		out=self.activation(out)

		return out

class StyleLoss(nn.Module):
	def __init__(self):
		super(StyleLoss, self).__init__()
		pass

	def forward(self, target_features, output_features):

		loss=0

		for target_f,out_f in zip(target_features, output_features):
			#target is batch size 1
			t_bs,t_ch,t_w,t_h=target_f.shape
			assert t_bs ==1, 'Network should be trained for only one target image'

			target_f=target_f.reshape(t_ch, t_w*t_h)
			
			target_gram_matrix=torch.matmul(target_f,target_f.T)/(t_ch*t_w*t_h) #t_ch x t_ch matrix

			i_bs, i_ch, i_w, i_h = out_f.shape

			assert t_ch == i_ch, 'Bug'

			for img_f in out_f: #contains features for batch of images
				img_f=img_f.reshape(i_ch, i_w*i_h)

				img_gram_matrix=torch.matmul(img_f, img_f.T)/(i_ch*i_w*i_h)

				loss+= torch.square(target_gram_matrix - img_gram_matrix).mean()

		return loss

class ContentLoss(nn.Module):
	def __init__(self):
		super(ContentLoss, self).__init__()

	def forward(self, style_features, content_features):
		loss=0
		for sf,cf in zip(style_features, content_features):
			a,b,c,d=sf.shape
			loss+=(torch.square(sf-cf)/(a*b*c*d)).mean()

		return loss

class TotalVariationLoss(nn.Module):
	def __init__(self):
		super(TotalVariationLoss, self).__init__()

	def forward(self, x):
		horizontal_loss=torch.pow(x[...,1:,:]-x[...,:-1,:],2).sum()

		vertical_loss=torch.pow(x[...,1:]-x[...,:-1],2).sum()

		return (horizontal_loss+vertical_loss)/x.numel()



class StyleTrainer(object):
	def __init__(self, student_network, loss_network, style_target_path, data_manager,feature_loss, style_loss, savepath=None):
		self.student_network=student_network
		self.loss_network=loss_network
		
		assert os.path.exists(style_target_path), 'Style target does not exist'
		image=Image.open(style_target_path).convert('RGB').resize(cfg.SIZE[::-1])
		preprocess=T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		self.style_target=torch.unsqueeze(preprocess(image),0)

		self.manager=data_manager

		self.feature_loss=feature_loss

		self.style_loss=style_loss

		self.total_variation = TotalVariationLoss()

		self.savepath=savepath

		self.writer=SummaryWriter()

		self.optimizer=optim.Adam(self.student_network.parameters(), lr=cfg.LR)

	def train(self, epochs=None, eval_interval=None, style_loss_weight=1.0):
		pass
		epochs= epochs if epochs else cfg.EPOCHS
		eval_interval=eval_interval if eval_interval else cfg.EVAL_INTERVAL

		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		train_loader, valid_loader, *_ = self.manager.dataloaders #ignore test loader if any

		self.student_network.to(device).train()
		
		self.loss_network.to(device)
		self.loss_network.eval()

		self.student_network, self.optimizer = amp.initialize(self.student_network, self.optimizer,
									opt_level='O2', enabled=True)

		self.style_target=self.style_target.to(device)

		style_target_features=resnet_forward(self.loss_network,self.style_target) #fixed during training

		step=0

		for epoch in range(epochs):
			estart=time.time()
			for x in train_loader:
				self.optimizer.zero_grad()

				x=x.to(device)

				stylized_image = self.student_network(x)
				content_features = resnet_forward(self.loss_network, x) #self.loss_network(x)
				stylized_features= resnet_forward(self.loss_network, stylized_image)#self.loss_network(stylized_image)

				feature_loss=self.feature_loss(stylized_features, content_features)

				style_loss=self.style_loss(style_target_features, content_features)

				tvloss=self.total_variation(stylized_image)

				loss = 1000*feature_loss + style_loss_weight*style_loss + 0.02*tvloss

				self.writer.add_scalar('Feature loss', feature_loss.item(), step)
				self.writer.add_scalar('Style loss', style_loss.item(), step)
				self.writer.add_scalar('Total Variation Loss', tvloss.item(), step)

				#loss.backward()
				with amp.scale_loss(loss, self.optimizer) as scaled_loss:
					scaled_loss.backward()

				self.optimizer.step()

				step+=1

				if step%eval_interval==0:
					self.student_network.eval()

					with torch.no_grad():
						pass
						for imgs in valid_loader:
							imgs=imgs.to(device)
							stylized=self.student_network(imgs)
							self.writer.add_images('Stylized Examples', stylized, step)
							break #just one batch is enough

					self.student_network.train()

			self.save(epoch)
			eend=time.time()
			print('Time taken for last epoch = {:.3f}'.format(eend-estart))

	def save(self, epoch):
		if self.savepath:
			path=self.savepath.format(epoch)
			torch.save(self.student_network.state_dict(), path)
			print(f'Saved model to {path}')

def resnet_forward(net, x):
	layers_used=['layer1', 'layer2', 'layer3', 'layer4']
	output=[]
	#print(net._modules.keys())
	for name, module in net._modules.items():
		if name=='fc':
			continue #dont run fc layer since _modules does not include flatten

		x=module(x)
		if name in layers_used:
			output.append(x)
	#print('Resnet forward method called')
	#[print(q.shape) for q in output]
	return output

if __name__=="__main__":
	net=StyleNetwork()
	manager=DataManager(cfg.IMGPATH_FILE, None, cfg.SIZE) #Datamanager without soft targets
	styleloss=StyleLoss()
	contentloss=ContentLoss()
	loss_network= models.resnet18()
	loss_network.load_state_dict(torch.load(cfg.LOSS_NET_PATH)['model'])

	for p in loss_network.parameters():
		p.requires_grad=False #freeze loss network

	trainer=StyleTrainer(net, loss_network,cfg.STYLE_TARGET, manager, contentloss, styleloss, './style_{}.pth')
	trainer.train()
