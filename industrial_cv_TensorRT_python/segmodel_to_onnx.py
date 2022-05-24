import torch
from torch import nn
from torchvision import models
import torchvision.transforms as T
import numpy as np
import cv2
import time
from segcolors import colors

class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net= models.segmentation.fcn_resnet50(pretrained=True, aux_loss=False).cuda()
        self.ppmean=torch.Tensor([0.485, 0.456, 0.406])
        self.ppstd=torch.Tensor([0.229, 0.224, 0.225])
        self.preprocessor=T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        self.cmap=torch.from_numpy(colors[:,::-1].copy())

    def forward(self, x):
        """x is a pytorch tensor"""

        #x=(x-self.ppmean)/self.ppstd #uncomment if you want onnx to include pre-processing
        isize=x.shape[-2:]
        x=self.net.backbone(x)['out']
        x=self.net.classifier(x)
        #x=nn.functional.interpolate(x, isize, mode='bilinear') #uncomment if you want onnx to include interpolation
        return x

    def export_onnx(self, onnxpath):
        """onnxpath: string, path of output onnx file"""

        x=torch.randn(1,3,360,640).cuda() #360p size
        input=['image']
        output=['probabilities']
        torch.onnx.export(self, x, onnxpath, verbose=False, input_names=input, output_names=output, opset_version=11)
        print('Exported to onnx')

    def infervideo(self, fname, view=True, savepath=None):
        """
        fname: path of input video file/camera index
        view(bool): whether or not to display results
        savepath (string or None): if path specified, output video is saved
        """
        src=cv2.VideoCapture(fname)
        ret,frame=src.read()

        if not ret:
            print(f'Cannot read input file/camera {fname}')
            quit()

        self.net.eval()

        dst=None
        fps=0.0

        if savepath is not None:
            dst=self.getvideowriter(savepath, src)

        with torch.no_grad(): #we just inferring, no need to calculate gradients
            while ret:
                outf, cfps=self.inferframe(frame, benchmark=True)
                if view:
                    cv2.imshow('segmentation', outf)
                    k=cv2.waitKey(1)
                    if k==ord('q'):
                        break
                if dst:
                    dst.write(outf)

                fps=0.9*fps+0.1*cfps
                print(fps)
                ret,frame=src.read()

            src.release()
            if dst:
                dst.release()

    def inferframe(self, frame, benchmark=True):
        """
        frame: numpy array containing un-pre-processed video frame (dtype is uint8)
        benchamrk: bool, whether or not to calculate inference time
        """
        rgb=frame[...,::-1].copy()
        processed=self.preprocessor(rgb)[None]
        start, end = 1e6, 0
        
        if benchmark:
            start=time.time()

        processed=processed.cuda() #transfer to GPU <-- does not use zero copy
        inferred= self(processed) #infer
        
        if benchmark:
            end=time.time()
        
        inferred=inferred.argmax(dim=1)

        overlaid=self.overlay(frame, inferred)
        return overlaid, 1.0/(end-start)

    def overlay(self, bgr, mask):
        """
        overlay pixel-wise predictions on input frame
        bgr: (numpy array) original video frame read from video/camera
        mask: (numpy array) class mask containing one of 21 classes for each pixel
        """
        colored = self.cmap[mask].to('cpu').numpy()[0,...]
        colored=cv2.resize(colored, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        oved = cv2.addWeighted(bgr, 0.7, colored, 0.3, 0.0)
        return oved

    def getvideowriter(self, savepath, srch):
        """
        Simple utility function for getting video writer
        savepath: string, path of output file
        src: a cv2.VideoCapture object
        """
        fps=srch.get(cv2.CAP_PROP_FPS)
        width=int(srch.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(srch.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc=int(srch.get(cv2.CAP_PROP_FOURCC))
        dst=cv2.VideoWriter(savepath, fourcc, fps, (width, height))
        return dst

if __name__=='__main__':
    model=SegModel()
    model.export_onnx('./segmodel.onnx')
    #model.infervideo('../may20/cam_2.mp4') #uncomment to infer on a video or camera


