import cv2
import numpy as np
import subprocess
import torch
from stylenet import StyleNetwork
from torchvision import transforms as T

net=StyleNetwork('./models/style_7.pth')

for p in net.parameters():
	p.requires_grad=False

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

net=net.eval().to(device) #use eval just for safety

src=cv2.VideoCapture('/dev/video0') #USB camera ID

ffstr='ffmpeg -re -f rawvideo -pix_fmt rgb24 -s 640x480 -i - -f v4l2 -pix_fmt yuv420p /dev/video2'
#ffmpeg pipeline which accepts raw rgb frames from command line and writes to virtul camera in yuv420p format

zoom=subprocess.Popen(ffstr, shell=True, stdin=subprocess.PIPE) #open process with shell so we can write to it

dummyframe=255*np.ones((480,640,3), dtype=np.uint8) #blank frame if camera cannot be read

preprocess=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
#same normalization as that used in training data

ret, frame=src.read()

while True:
	try:
		if ret:
			frame=(frame[:,:,::-1]/255.0).astype(np.float32) #convert BGR to RGB, convert to 0-1 range and cast to float32
			frame_tensor=torch.unsqueeze(torch.from_numpy(frame),0).permute(0,3,1,2)
			# add batch dimension and convert to NCHW format

			tensor_in = preprocess(frame_tensor) #normalize
			tensor_in=tensor_in.to(device) #send to GPU

			tensor_out = net(tensor_in) #stylized tensor
			tensor_out=torch.squeeze(tensor_out).permute(1,2,0) #remove batch dimension and convert to HWC (opencv format)
			stylized_frame=(255*(tensor_out.to('cpu').detach().numpy())).astype(np.uint8) #convert to 0-255 range and cast as uint8
			#gaussian_blur = cv2.GaussianBlur(stylized_frame, (0, 0), 2.0)
			#stylized_frame = cv2.addWeighted(stylized_frame, 1.5, gaussian_blur, -0.5, 0, stylized_frame)
		else:
			stylized_frame=dummyframe #if camera cannot be read, blank white image will be shown

		zoom.stdin.write(stylized_frame.tobytes()) 
		#write to ffmpeg pipeline which in turn writes to virtual camera that can be accessed by zoom/skype/teams

		ret,frame=src.read()
	except KeyboardInterrupt:
		print('Received stop command')
		break

zoom.terminate()
src.release() #close ffmpeg pipeline and release camera