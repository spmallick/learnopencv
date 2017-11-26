import cv2
import numpy as np
import sys


def readImagesAndTimes():
  
  filenames = [
               "images/memorial0061.jpg",
               "images/memorial0062.jpg",
               "images/memorial0063.jpg",
               "images/memorial0064.jpg",
               "images/memorial0065.jpg",
               "images/memorial0066.jpg",
               "images/memorial0067.jpg",
               "images/memorial0068.jpg",
               "images/memorial0069.jpg",
               "images/memorial0070.jpg",
               "images/memorial0071.jpg",
               "images/memorial0072.jpg",
               "images/memorial0073.jpg",
               "images/memorial0074.jpg",
               "images/memorial0075.jpg",
               "images/memorial0076.jpg"
               ]

  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
  
  return images

if __name__ == '__main__':
  
  # Read images
  print("Reading images ... ")
  
  if len(sys.argv) > 1:
    # Read images from the command line
    images = []
    for filename in sys.argv[1:]:
      im = cv2.imread(filename)
      images.append(im)
    needsAlignment = False
  else :
    # Read example images
    images = readImagesAndTimes()
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


