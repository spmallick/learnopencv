from profiler_demo_utils import *
#importing * is not good practice, but simplifies
#this demo. Please do not imitate this :-)

class VisionTrainer(object):
	def __init__(self, net, dm):
		pass
		self.net=net
		self.dm=dm
		self.writer=SummaryWriter()
		self.criterion=nn.CrossEntropyLoss()
		self.optimizer=optim.AdamW(self.net.parameters(), lr=1e-6)
		self.savepath=None

	def train(self, epochs, save, profiler=None):
		pass
		eval_interval=200 #evaluate every 200 steps
		self.savepath=save
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		train_loader, valid_loader = self.dm.train_loader, self.dm.valid_loader #ignore test loader if any

		self.net.to(device).train()
	
		if has_apex:
			self.net, self.optimizer = amp.initialize(self.net, self.optimizer,
										opt_level='O2', enabled=True)

		step=0
		
		get_accuracy=lambda p,y: (torch.argmax(p, dim=1) == y).to(torch.float).mean().item()

		for epoch in range(epochs):
			estart=time.time()
			for x,y in train_loader:
				with record_function("training_events"): #record these as training_events
					self.optimizer.zero_grad()

					x=x.to(device)
					y=y.to(device)
					
					pred = self.net(x)
					
					loss = self.criterion(pred,y)

					#print(loss.item())
					self.writer.add_scalar('Training Loss', loss.item(), step)

					with amp.scale_loss(loss, self.optimizer) as scaled_loss:
						scaled_loss.backward()

					#loss.backward()

					torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
					self.optimizer.step()
					acc=get_accuracy(pred, y)
					step+=1
					self.writer.add_scalar('Training Accuracy', acc, step)

				if step%eval_interval==0:
					with record_function("evaluation_events"): #record these as evaluation_events
						self.net.eval()
						valoss=[]
						vaacc=[]
						with torch.no_grad():
							pass
							for imgs, ys in valid_loader:
								imgs=imgs.to(device)
								ys=ys.to(device)
								preds=self.net(imgs)
								vacc=get_accuracy(preds, ys)
								vloss=self.criterion(preds, ys)
								#pdb.set_trace()
								valoss.append(vloss.flatten().item())
								vaacc.append(vacc)

						self.writer.add_scalar('Validation Loss', np.mean(valoss), step)
						self.writer.add_scalar('Validation Accuracy', np.mean(vaacc), step)
						self.net.train()

				if profiler:
					profiler.step()

			self.save(epoch)
			eend=time.time()
			print('Time taken for last epoch = {:.3f}'.format(eend-estart))

	def save(self, epoch):
		if self.savepath:
			path=self.savepath.format(epoch)
			torch.save(self.net.state_dict(), path)
			print(f'Saved model to {path}')

		
def main():
	dm=CIFAR10_Manager('./cf10')

	#Just change name to one of the following:
	#resnet18, resnet50, mobilenetv3, densenet, squeezenet, inception
	mname='resnet50'
	net=VisionClassifier(nclasses=10, mname=mname)

	trainer=VisionTrainer(net,dm)
	
	with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
		record_shapes=True,
		schedule=schedule(
        wait=1,
        warmup=1,
        active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs'),
		profile_memory=True, 
		use_cuda=True) as prof:

		trainer.train(epochs=1, save='models/cf10_{}.pth', profiler=prof)

if __name__=='__main__':
	main()
