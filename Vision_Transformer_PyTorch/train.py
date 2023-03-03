import darklight as dl
import torch
from vit import VisionTransformer
import vitconfigs as vcfg

net=VisionTransformer(vcfg.base)
dm=dl.ImageNetManager('/sfnvme/imagenet/', size=[224,224], bsize=128)

opt_params={
	'optimizer': torch.optim.AdamW,
	'okwargs': {'lr': 1e-4, 'weight_decay':0.05},
	'scheduler':torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
	'skwargs': {'T_0':10,'T_mult':2},
	'amplevel': None
	}
trainer=dl.StudentTrainer(net, dm, None, opt_params=opt_params)
trainer.train(epochs=300, save='vitbase_{}.pth')