import pyvirtualcam
import numpy as np
import cv2
import torch
from stylenet import StyleNetwork
from torchvision import transforms as T


net=StyleNetwork('./style_7.pth')

for p in net.parameters():
    p.requires_grad=False

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

net=net.eval().to(device) #use eval just for safety

src=cv2.VideoCapture(1) #USB camera ID

dummyframe=127*np.ones((720,1280,3),np.uint8)

preprocess=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
#same normalization as that used in training data

ret, frame=src.read()

with pyvirtualcam.Camera(width=1280, height=720, fps=30) as vcam:
    
    print(f'Using virtual camera: {vcam.device} at {vcam.width} x {vcam.height}')

    while True:
        try:
            if ret:
                frame=cv2.resize(frame,(640,480))
                frame=(frame[:,:,::-1]/255.0).astype(np.float32) #convert BGR to RGB, convert to 0-1 range and cast to float32
                frame_tensor=torch.unsqueeze(torch.from_numpy(frame),0).permute(0,3,1,2)
                # add batch dimension and convert to NCHW format

                tensor_in = preprocess(frame_tensor) #normalize
                tensor_in=tensor_in.to(device) #send to GPU

                tensor_out = net(tensor_in) #stylized tensor
                tensor_out=torch.squeeze(tensor_out).permute(1,2,0) #remove batch dimension and convert to HWC (opencv format)
                stylized_frame=(255*(tensor_out.to('cpu').detach().numpy())).astype(np.uint8) #convert to 0-255 range and cast as uint8
                stylized_frame=cv2.resize(stylized_frame, (1280, 720))
            else:
                stylized_frame=dummyframe #if camera cannot be read, blank white image will be shown

            vcam.send(stylized_frame)
            #write to ffmpeg pipeline which in turn writes to virtual camera that can be accessed by zoom/skype/teams

            ret,frame=src.read()

        except KeyboardInterrupt:
            print('Received stop command')
            break

src.release() #close ffmpeg pipeline and release camera
