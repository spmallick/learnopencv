#include <opencv2/photo.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

// Read Images
void readImages(vector<Mat> &images)
{
  
  int numImages = 16;
  static const char* filenames[] =
  {
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
  };
  
  for(int i=0; i < numImages; i++)
  {
    Mat im = imread(filenames[i]);
    images.push_back(im);
  }

}

int main(int argc, char **argv)
{
  // Read images
  cout << "Reading images ... " << endl;
  vector<Mat> images;
  
  bool needsAlignment = true;
  if(argc > 1)
  {
    // Read images from the command line
    for(int i=1; i < argc; i++)
    {
      Mat im = imread(argv[i]);
      images.push_back(im);
    }

  }
  else
  {
    // Read example images
    readImages(images);
    needsAlignment = false;
  }
  
  // Align input images
  if(needsAlignment)
  {
    cout << "Aligning images ... " << endl;
    Ptr<AlignMTB> alignMTB = createAlignMTB();
    alignMTB->process(images, images);
  }
  else
  {
    cout << "Skipping alignment ... " << endl;
  }
  
  
  // Merge using Exposure Fusion
  cout << "Merging using Exposure Fusion ... " << endl;
  Mat exposureFusion;
  Ptr<MergeMertens> mergeMertens = createMergeMertens();
  mergeMertens->process(images, exposureFusion);
  
  // Save output image
  cout << "Saving output ... exposure-fusion.jpg"<< endl;
  imwrite("exposure-fusion.jpg", exposureFusion * 255);
  
  return EXIT_SUCCESS;
}
