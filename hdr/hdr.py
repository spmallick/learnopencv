import cv2
import numpy as np
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
  # Read images and exposure times
  print("Reading images ... ")

  dir_path = '' #path to images
  images, times = readImagesAndTimes(dir_path=dir_path)
  
  # Align input images
  print("Aligning images ... ")
  alignMTB = cv2.createAlignMTB()
  alignMTB.process(images, images)
  
  # Obtain Camera Response Function (CRF)
  print("Calculating Camera Response Function (CRF) ... ")
  calibrateDebevec = cv2.createCalibrateDebevec()
  responseDebevec = calibrateDebevec.process(images, times)
  
  # Merge images into an HDR linear image
  print("Merging images into one HDR image ... ")
  mergeDebevec = cv2.createMergeDebevec()
  hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
  # Save HDR image.
  cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
  print("saved hdrDebevec.hdr ")
  
  # Tonemap using Drago's method to obtain 24-bit color image
  print("Tonemaping using Drago's method ... ")
  tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
  ldrDrago = tonemapDrago.process(hdrDebevec)
  ldrDrago = 3 * ldrDrago
  cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)
  print("saved ldr-Drago.jpg")
  
  # Tonemap using Durand's method obtain 24-bit color image
  print("Tonemaping using Durand's method ... ")
  tonemapDurand = cv2.createTonemapDurand(1.5,4,1.0,1,1)
  ldrDurand = tonemapDurand.process(hdrDebevec)
  ldrDurand = 3 * ldrDurand
  cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)
  print("saved ldr-Durand.jpg")
  
  # Tonemap using Reinhard's method to obtain 24-bit color image
  print("Tonemaping using Reinhard's method ... ")
  tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
  ldrReinhard = tonemapReinhard.process(hdrDebevec)
  cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
  print("saved ldr-Reinhard.jpg")
  
  # Tonemap using Mantiuk's method to obtain 24-bit color image
  print("Tonemaping using Mantiuk's method ... ")
  tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
  ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
  ldrMantiuk = 3 * ldrMantiuk
  cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
  print("saved ldr-Mantiuk.jpg")

