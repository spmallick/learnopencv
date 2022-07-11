
import numpy as np

from math import sin, cos


def find_T_matrix(pts,t_pts):
	A = np.zeros((8,9))
	for i in range(0,4):
		xi  = pts[:,i];
		xil = t_pts[:,i];
		xi  = xi.T
		
		A[i*2,   3:6] = -xil[2]*xi
		A[i*2,   6: ] =  xil[1]*xi
		A[i*2+1,  :3] =  xil[2]*xi
		A[i*2+1, 6: ] = -xil[0]*xi

	
	[U,S,V] = np.linalg.svd(A)
	H = V[-1,:].reshape((3,3))

	return H

def getRectPts(tlx,tly,brx,bry):
	return np.matrix([[tlx,brx,brx,tlx],[tly,tly,bry,bry],[1.,1.,1.,1.]],dtype=float)

def perspective_transform(wh,angles=np.array([0.,0.,0.]),zcop=1000., dpp=1000.):
	rads = np.deg2rad(angles)

	a = rads[0]; Rx = np.matrix([[1, 0, 0]				, [0, cos(a), sin(a)]	, [0, -sin(a), cos(a)]	])
	a = rads[1]; Ry = np.matrix([[cos(a), 0, -sin(a)]	, [0, 1, 0]				, [sin(a), 0, cos(a)]	])
	a = rads[2]; Rz = np.matrix([[cos(a), sin(a), 0]	, [-sin(a), cos(a), 0]	, [0, 0, 1]				])

	R = Rx*Ry*Rz;

	(w,h) = tuple(wh)
	xyz = np.matrix([[0,0,w,w],[0,h,0,h],[0,0,0,0]])
	hxy = np.matrix([[0,0,w,w],[0,h,0,h],[1,1,1,1]])

	xyz = xyz - np.matrix([[w],[h],[0]])/2.
	xyz = R*xyz

	xyz = xyz - np.matrix([[0],[0],[zcop]])
	hxyz = np.concatenate([xyz,np.ones((1,4))])

	P = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,-1./dpp,0]])
	_hxy = P*hxyz
	_hxy = _hxy/_hxy[2,:]
	_hxy = _hxy + np.matrix([[w],[h],[0]])/2.
	
	return find_T_matrix(hxy,_hxy)