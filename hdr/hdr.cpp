#include <opencv2/photo.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;


void readImagesAndTimes(vector<Mat> &images, vector<float> &times)
{
  
  int numImages = 4;
  
  static const float timesArray[] = {1/30.0f,0.25,2.5,15.0};
  times.assign(timesArray, timesArray + numImages);
  
  static const char* filenames[] = {"img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"};
  for(int i=0; i < numImages; i++)
  {
    Mat im = imread(filenames[i]);
    images.push_back(im);
  }

}

int main(int, char**argv)
{
  // Read images and exposure times
  cout << "Reading images ... " << endl;
  vector<Mat> images;
  vector<float> times;
  readImagesAndTimes(images, times);
  
  
  // Align input images
  cout << "Aligning images ... " << endl;
  Ptr<AlignMTB> alignMTB = createAlignMTB();
  alignMTB->process(images, images);
  
  // Obtain Camera Response Function (CRF)
  cout << "Calculating Camera Response Function (CRF) ... " << endl;
  Mat responseDebevec;
  Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
  calibrateDebevec->process(images, responseDebevec, times);
  
  // Merge images into an HDR linear image
  cout << "Merging images into one HDR image ... " ;
  Mat hdrDebevec;
  Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
  mergeDebevec->process(images, hdrDebevec, times, responseDebevec);
  // Save HDR image.
  imwrite("hdrDebevec.hdr", hdrDebevec);
  cout << "saved hdrDebevec.hdr "<< endl;
  
  // Tonemap using Drago's method to obtain 24-bit color image
  cout << "Tonemaping using Drago's method ... ";
  Mat ldrDrago;
  Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0, 0.7);
  tonemapDrago->process(hdrDebevec, ldrDrago);
  ldrDrago = 3 * ldrDrago;
  imwrite("ldr-Drago.jpg", ldrDrago * 255);
  cout << "saved ldr-Drago.jpg"<< endl;
  
  // Tonemap using Durand's method obtain 24-bit color image
  cout << "Tonemaping using Durand's method ... ";
  Mat ldrDurand;
  Ptr<TonemapDurand> tonemapDurand = createTonemapDurand(1.5,4,1.0,1,1);
  tonemapDurand->process(hdrDebevec, ldrDurand);
  ldrDurand = 3 * ldrDurand;
  imwrite("ldr-Durand.jpg", ldrDurand * 255);
  cout << "saved ldr-Durand.jpg"<< endl;
  
  // Tonemap using Reinhard's method to obtain 24-bit color image
  cout << "Tonemaping using Reinhard's method ... ";
  Mat ldrReinhard;
  Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5, 0,0,0);
  tonemapReinhard->process(hdrDebevec, ldrReinhard);
  imwrite("ldr-Reinhard.jpg", ldrReinhard * 255);
  cout << "saved ldr-Reinhard.jpg"<< endl;
  
  // Tonemap using Mantiuk's method to obtain 24-bit color image
  cout << "Tonemaping using Mantiuk's method ... ";
  Mat ldrMantiuk;
  Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2,0.85, 1.2);
  tonemapMantiuk->process(hdrDebevec, ldrMantiuk);
  ldrMantiuk = 3 * ldrMantiuk;
  imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255);
  cout << "saved ldr-Mantiuk.jpg"<< endl;
  
  
  
  return EXIT_SUCCESS;
}
