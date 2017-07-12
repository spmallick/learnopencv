// Include OpenCV header
#include <opencv2/opencv.hpp>

// Use cv and std namespaces
using namespace cv;
using namespace std;

// Define a pixel
typedef Point3_<uint8_t> Pixel;

// tic is called to start timer
void tic(double &t)
{
  t = (double)getTickCount();
}

// toc is called to end timer
double toc(double &t)
{
  return ((double)getTickCount() - t)/getTickFrequency();
}

void complicatedThreshold(Pixel &pixel)
{
  if (pow(double(pixel.x)/10,2.5) > 100)
  {
    pixel.x = 255;
    pixel.y = 255;
    pixel.z = 255;
  }
  else
  {
    pixel.x = 0;
    pixel.y = 0;
    pixel.z = 0;
  }
}



// Parallel execution with function object.
struct Operator
{
  void operator ()(Pixel &pixel, const int * position) const
  {
    // Perform a simple threshold operation
   	complicatedThreshold(pixel);
  }
};


int main(int argc, char** argv)
{
  // Read image
  Mat image = imread("butterfly.jpg");
  
  // Scale image 30x
  resize(image,image, Size(), 30, 30);
  
  // Print image size
  cout << "Image size " << image.size() << endl;
  
  // Number of trials
  int numTrials = 5;
  
  // Print number of trials
  cout << "Number of trials : " << numTrials << endl;
  
  // Make two copies
  Mat image1 = image.clone();
  Mat image2 = image.clone();
  Mat image3 = image.clone();
  
  // Start timer
  double t;
  tic(t);
  
  for (int n = 0; n < numTrials; n++)
  {
    // Naive pixel access
    // Loop over all rows
    for (int r = 0; r < image.rows; r++)
    {
      // Loop over all columns
      for ( int c = 0; c < image.cols; c++)
      {
        // Obtain pixel at (r, c)
        Pixel pixel = image.at<Pixel>(r, c);
        // Apply complicatedTreshold
        complicatedThreshold(pixel);
        // Put result back
        image.at<Pixel>(r, c) = pixel;
      }
      
    }
  }
  
  cout << "Naive way: " << toc(t) << endl;
  
  
  // Start timer
  tic(t);
  
  // image1 is guaranteed to be continous, but
  // if you are curious uncomment the line below
  // cout << "Image 1 is continous : " << image1.isContinuous() << endl;
  
  for (int n = 0; n < numTrials; n++)
  {
    // Get pointer to first pixel
    Pixel* pixel = image1.ptr<Pixel>(0,0);
    
    // Mat objects created using the create method are stored
    // in one continous memory block.
    const Pixel* endPixel = pixel + image1.cols * image1.rows;
    
    // Loop over all pixels
    for (; pixel != endPixel; pixel++)
    {
      complicatedThreshold(*pixel);
    }
    
    
  }
  cout << "Pointer Arithmetic " << toc(t) << endl;
  tic(t);
  
  for (int n = 0; n < numTrials; n++)
  {
    image2.forEach<Pixel>(Operator());
  }
  cout << "forEach : " << toc(t) << endl;
  
#if __cplusplus >= 201103L || (__cplusplus < 200000 && __cplusplus > 199711L)
  tic(t);
  
  for (int n = 0; n < numTrials; n++)
  {
    // Parallel execution using C++11 lambda.
    image3.forEach<Pixel>
    (
      [](Pixel &pixel, const int * position) -> void
      {
        complicatedThreshold(pixel);
      }
     );
  }
  cout << "forEach C++11 : " << toc(t) << endl;
  
#endif
  
  return EXIT_SUCCESS;
}

