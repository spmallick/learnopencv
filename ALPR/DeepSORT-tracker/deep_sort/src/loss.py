
import tensorflow as tf


def logloss(Ptrue,Pred,szs,eps=10e-10):
	b,h,w,ch = szs
	Pred = tf.clip_by_value(Pred,eps,1.)
	Pred = -tf.log(Pred)
	Pred = Pred*Ptrue
	Pred = tf.reshape(Pred,(b,h*w*ch))
	Pred = tf.reduce_sum(Pred,1)
	return Pred

def l1(true,pred,szs):
	b,h,w,ch = szs
	res = tf.reshape(true-pred,(b,h*w*ch))
	res = tf.abs(res)
	res = tf.reduce_sum(res,1)
	return res

def loss(Ytrue, Ypred):

	b = tf.shape(Ytrue)[0]
	h = tf.shape(Ytrue)[1]
	w = tf.shape(Ytrue)[2]

	obj_probs_true = Ytrue[...,0]
	obj_probs_pred = Ypred[...,0]

	non_obj_probs_true = 1. - Ytrue[...,0]
	non_obj_probs_pred = Ypred[...,1]

	affine_pred	= Ypred[...,2:]
	pts_true 	= Ytrue[...,1:]

	affinex = tf.stack([tf.maximum(affine_pred[...,0],0.),affine_pred[...,1],affine_pred[...,2]],3)
	affiney = tf.stack([affine_pred[...,3],tf.maximum(affine_pred[...,4],0.),affine_pred[...,5]],3)

	v = 0.5
	base = tf.stack([[[[-v,-v,1., v,-v,1., v,v,1., -v,v,1.]]]])
	base = tf.tile(base,tf.stack([b,h,w,1]))

	pts = tf.zeros((b,h,w,0))

	for i in range(0,12,3):
		row = base[...,i:(i+3)]
		ptsx = tf.reduce_sum(affinex*row,3)
		ptsy = tf.reduce_sum(affiney*row,3)

		pts_xy = tf.stack([ptsx,ptsy],3)
		pts = (tf.concat([pts,pts_xy],3))

	flags = tf.reshape(obj_probs_true,(b,h,w,1))
	res   = 1.*l1(pts_true*flags,pts*flags,(b,h,w,4*2))
	res  += 1.*logloss(obj_probs_true,obj_probs_pred,(b,h,w,1))
	res  += 1.*logloss(non_obj_probs_true,non_obj_probs_pred,(b,h,w,1))
	return res