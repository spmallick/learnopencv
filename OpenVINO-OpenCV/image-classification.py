import numpy as np
import time
import cv2

caffe_root = '/home/ubuntu/caffe/'
image = cv2.imread('/home/ubuntu/caffe/examples/images/cat.jpg')
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
prototxt = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

// load the labels file
rows = open(labels_file).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

blob = cv2.dnn.blobFromImage(image,1,(224,224),(104,117,123))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt,model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# set the blob as input to the network and perform a forward-pass to
# obtain our output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took " + str((end-start)*1000) + " ms")
