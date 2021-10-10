import torchvision.models as models
from dataset import DataManager
from PIL import Image
import torch
import glob
import numpy as np
import time
import config as cfg

manager=DataManager(cfg.IMGPATH_FILE, cfg.SOFT_TARGET_PATH, [224,224])

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('Using device {}'.format(torch.cuda.get_device_name(device)))

BATCH_SIZE=256

results_152=np.zeros((n_images, 1000), dtype=np.float32)

resnet152 = models.resnet152(pretrained=True, progress=True).to(device).eval()

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

begin=time.time()

for start in range(0,n_images, BATCH_SIZE):
	end= min(start+BATCH_SIZE, n_images)
	batch_names=image_paths[start:end]
	batch_images=[Image.open(p).convert('RGB') for p in batch_names]
	
	with torch.no_grad():
		tensor_images=[torch.unsqueeze(transform(img),0) for img in batch_images]
		in_tensor=torch.cat(tensor_images).to(device)
		
		out_tensor_152=resnet152(in_tensor)
		
		out_numpy_152=out_tensor_152.cpu().detach().numpy()
		
	results_152[start:end,:]=out_numpy_152

	pg=100*end/n_images
	sys.stdout.write('\r Progress= {:.2f} %'.format(pg))

np.save(f'resnet152_results.npy', results_152)

end=time.time()
print('Total time taken for inference = {:.2f}'.format(end-begin))
