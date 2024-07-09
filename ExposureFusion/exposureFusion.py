import cv2
import numpy as np
import sys
import os

def readImagesAndTimes(dir_path):
  
  #load images with os
  filenames = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]
  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    im = cv2.resize(im, (500, 500))
    images.append(im)
  
  return images

if __name__ == '__main__':
  
  # Read images
  print("Reading images ... ")
  
  dir_path = '' #path to images
  
  if len(sys.argv) > 1:
    # Read images from the command line
    images = []
    for filename in sys.argv[1:]:
      im = cv2.imread(filename)
      images.append(im)
    needsAlignment = False
  else :
    # Read example images
    images = readImagesAndTimes(dir_path=dir_path)
    needsAlignment = False
  
  # Align input images
  if needsAlignment:
    print("Aligning images ... ")
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
  else :
    print("Skipping alignment ... ")
  
  # Merge using Exposure Fusion
  print("Merging using Exposure Fusion ... ");
  mergeMertens = cv2.createMergeMertens()
  exposureFusion = mergeMertens.process(images)

  # Save output image
  print("Saving output ... exposure-fusion.jpg")
  cv2.imwrite("exposure-fusion.jpg", exposureFusion * 255)


