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
  
  static const float timesArray[] = {1/30.0f,0.25,2.5,15};
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
  vector<Mat> images;
  vector<float> times;
  readImagesAndTimes(images, times);
  
  // Obtain Camera Response Function (CRF)
  Mat responseDebevec;
  Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
  calibrateDebevec->process(images, responseDebevec, times);
  
  // Merge images into one HDR image
  Mat hdrDebevec;
  Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
  mergeDebevec->process(images, hdrDebevec, times, responseDebevec);
  
  // Tonemap HDR images to obtain 24-bit color image
  Mat ldr;
  Ptr<TonemapDurand> tonemapDurand = createTonemapDurand(2.2f);
  tonemapDurand->process(hdrDebevec, ldr);
  
  // Alternative method that does not require intermediate HDR image
  Mat fusion;
  Ptr<MergeMertens> mergeMertens = createMergeMertens();
  mergeMertens->process(images, fusion);
  
  // Write results
  imwrite("fusion.jpg", fusion * 255);
  imwrite("ldr-Debevec-Durand.jpg", ldr * 255);
  imwrite("hdrDebevec.hdr", hdrDebevec);
  
  return EXIT_SUCCESS;
}
