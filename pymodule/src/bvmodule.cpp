#include"bvmodule.hpp"

namespace bv
{

  void fillHoles(Mat &im)
  {
    Mat im_th;

    // Binarize the image by thresholding
    threshold(im, im_th, 128, 255, THRESH_BINARY);
    // Flood fill
    Mat im_floodfill = im_th.clone();
    floodFill(im_floodfill, cv::Point(0,0), Scalar(255));

    // Invert floodfilled image
    Mat im_floodfill_inv;
    bitwise_not(im_floodfill, im_floodfill_inv);

    // Combine the two images to fill holes
    im = (im_th | im_floodfill_inv);

  }


  void Filters::edge(InputArray im, OutputArray imedge) 
  {
    // Perform canny edge detection
    Canny(im,imedge,100,200); 
  }

  Filters::Filters() 
  {
  }
}

