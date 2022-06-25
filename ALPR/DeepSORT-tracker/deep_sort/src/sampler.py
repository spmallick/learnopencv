
import cv2
import numpy as np
import random

from src.utils 	import im2single, getWH, hsv_transform, IOU_centre_and_dims
from src.label	import Label
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts


def labels2output_map(label,lppts,dim,stride):

	side = ((float(dim) + 40.)/2.)/stride # 7.75 when dim = 208 and stride = 16

	outsize = dim/stride
	Y  = np.zeros((outsize,outsize,2*4+1),dtype='float32')
	MN = np.array([outsize,outsize])
	WH = np.array([dim,dim],dtype=float)

	tlx,tly = np.floor(np.maximum(label.tl(),0.)*MN).astype(int).tolist()
	brx,bry = np.ceil (np.minimum(label.br(),1.)*MN).astype(int).tolist()

	for x in range(tlx,brx):
		for y in range(tly,bry):

			mn = np.array([float(x) + .5, float(y) + .5])
			iou = IOU_centre_and_dims(mn/MN,label.wh(),label.cc(),label.wh())

			if iou > .5:

				p_WH = lppts*WH.reshape((2,1))
				p_MN = p_WH/stride

				p_MN_center_mn = p_MN - mn.reshape((2,1))

				p_side = p_MN_center_mn/side

				Y[y,x,0] = 1.
				Y[y,x,1:] = p_side.T.flatten()

	return Y

def pts2ptsh(pts):
	return np.matrix(np.concatenate((pts,np.ones((1,pts.shape[1]))),0))

def project(I,T,pts,dim):
	ptsh 	= np.matrix(np.concatenate((pts,np.ones((1,4))),0))
	ptsh 	= np.matmul(T,ptsh)
	ptsh 	= ptsh/ptsh[2]
	ptsret  = ptsh[:2]
	ptsret  = ptsret/dim
	Iroi = cv2.warpPerspective(I,T,(dim,dim),borderValue=.0,flags=cv2.INTER_LINEAR)
	return Iroi,ptsret

def flip_image_and_pts(I,pts):
	I = cv2.flip(I,1)
	pts[0] = 1. - pts[0]
	idx = [1,0,3,2]
	pts = pts[...,idx]
	return I,pts

def augment_sample(I,pts,dim):

	maxsum,maxangle = 120,np.array([80.,80.,45.])
	angles = np.random.rand(3)*maxangle
	if angles.sum() > maxsum:
		angles = (angles/angles.sum())*(maxangle/maxangle.sum())

	I = im2single(I)
	iwh = getWH(I.shape)

	whratio = random.uniform(2.,4.)
	wsiz = random.uniform(dim*.2,dim*1.)
	
	hsiz = wsiz/whratio

	dx = random.uniform(0.,dim - wsiz)
	dy = random.uniform(0.,dim - hsiz)

	pph = getRectPts(dx,dy,dx+wsiz,dy+hsiz)
	pts = pts*iwh.reshape((2,1))
	T = find_T_matrix(pts2ptsh(pts),pph)

	H = perspective_transform((dim,dim),angles=angles)
	H = np.matmul(H,T)

	Iroi,pts = project(I,H,pts,dim)
	
	hsv_mod = np.random.rand(3).astype('float32')
	hsv_mod = (hsv_mod - .5)*.3
	hsv_mod[0] *= 360
	Iroi = hsv_transform(Iroi,hsv_mod)
	Iroi = np.clip(Iroi,0.,1.)

	pts = np.array(pts)

	if random.random() > .5:
		Iroi,pts = flip_image_and_pts(Iroi,pts)

	tl,br = pts.min(1),pts.max(1)
	llp = Label(0,tl,br)

	return Iroi,llp,pts
