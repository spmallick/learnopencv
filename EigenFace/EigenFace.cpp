#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;

#define MAX_SLIDER_VALUE 255
#define NUM_EIGEN_FACES 10


// Weights for the different eigenvectors
int sliderValues[NUM_EIGEN_FACES];

// Matrices for average (mean) and eigenvectors
Mat averageFace;
vector<Mat> eigenFaces;

// Read jpg files from the directory
void readImages(string dirName, vector<Mat> &images)
{
  
  cout << "Reading images from " << dirName;

  // Add slash to directory name if missing
  if (!dirName.empty() && dirName.back() != '/')
    dirName += '/';
  
  DIR *dir;
  struct dirent *ent;
  int count = 0;
  
  //image extensions
  string imgExt = "jpg";
  vector<string> files;
  
  if ((dir = opendir (dirName.c_str())) != NULL)
  {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL)
    {
      if(strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name,"..") == 0 )
      {
        continue;
      }
      string fname = ent->d_name;
      
      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos)
      {
        string path = dirName + fname;
        Mat img = imread(path);
        if(!img.data)
        {
          cout << "image " << path << " not read properly" << endl;
        }
        else
        { 
          // Convert images to floating point type
          img.convertTo(img, CV_32FC3, 1/255.0);
          images.push_back(img);
          
          // A vertically flipped image is also a valid face image.
          // So lets use them as well.
          Mat imgFlip;
          flip(img, imgFlip, 1);
          images.push_back(imgFlip);
        }
      }
    }
    closedir (dir);
  }

  // Exit program if no images are found
  if(images.empty())exit(EXIT_FAILURE);
  
  cout << "... " << images.size() / 2 << " files read"<< endl;
  
}

// Create data matrix from a vector of images
static  Mat createDataMatrix(const vector<Mat> &images)
{
  cout << "Creating data matrix from images ...";
  
  // Allocate space for all images in one data matrix.
  // The size of the data matrix is
  //
  // ( w  * h  * 3, numImages )
  //
  // where,
  //
  // w = width of an image in the dataset.
  // h = height of an image in the dataset.
  // 3 is for the 3 color channels.
  
  
  Mat data(static_cast<int>(images.size()), images[0].rows * images[0].cols * 3, CV_32F);
  
  // Turn an image into one row vector in the data matrix
  for(unsigned int i = 0; i < images.size(); i++)
  {
    // Extract image as one long vector of size w x h x 3
    Mat image = images[i].reshape(1,1);
    
    // Copy the long vector into one row of the destm
    image.copyTo(data.row(i));
    
  }
  
  cout << " DONE" << endl;
  return data;
}

// Calculate final image by adding weighted
// EigenFaces to the average face.
void createNewFace(int ,void *)
{
  // Start with the mean image
  Mat output = averageFace.clone();
  
  // Add the eigen faces with the weights
  for(int i = 0; i < NUM_EIGEN_FACES; i++)
  {
    // OpenCV does not allow slider values to be negative.
    // So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
    double weight = sliderValues[i] - MAX_SLIDER_VALUE/2;
    output = output + eigenFaces[i] * weight;
  }

  resize(output, output, Size(), 2, 2);
  
  imshow("Result", output);
  
}

// Reset slider values
void resetSliderValues(int event, int x, int y, int flags, void* userdata)
{
  if (event == EVENT_LBUTTONDOWN)
  {
    for(int i = 0; i < NUM_EIGEN_FACES; i++)
    {
      sliderValues[i] = 128;
      setTrackbarPos("Weight" + to_string(i), "Trackbars", MAX_SLIDER_VALUE/2);
    }
    
    createNewFace(0,0);
    
  }
}


int main(int argc, char **argv)
{
  // Directory containing images
  string dirName = "images/";
  
  // Read images in the directory
  vector<Mat> images; 
  readImages(dirName, images);
  
  // Size of images. All images should be the same size. 
  Size sz = images[0].size(); 
  
  // Create data matrix for PCA.
  Mat data = createDataMatrix(images);
  
  // Calculate PCA of the data matrix
  cout << "Calculating PCA ...";
  PCA pca(data, Mat(), PCA::DATA_AS_ROW, NUM_EIGEN_FACES);
  cout << " DONE"<< endl;
  
  // Extract mean vector and reshape it to obtain average face
  averageFace = pca.mean.reshape(3,sz.height);
  
  // Find eigen vectors.
  Mat eigenVectors = pca.eigenvectors;
  
  // Reshape Eigenvectors to obtain EigenFaces
  for(int i = 0; i < NUM_EIGEN_FACES; i++)
  {
      Mat eigenFace = eigenVectors.row(i).reshape(3,sz.height);
      eigenFaces.push_back(eigenFace);
  }
  
  // Show mean face image at 2x the original size
  Mat output;
  resize(averageFace, output, Size(), 2, 2);
  
  namedWindow("Result", CV_WINDOW_AUTOSIZE);
  imshow("Result", output);
  
  // Create trackbars
  namedWindow("Trackbars", CV_WINDOW_AUTOSIZE);
  for(int i = 0; i < NUM_EIGEN_FACES; i++)
  {
    sliderValues[i] = MAX_SLIDER_VALUE/2;
    createTrackbar( "Weight" + to_string(i), "Trackbars", &sliderValues[i], MAX_SLIDER_VALUE, createNewFace);
  }
  
  // You can reset the sliders by clicking on the mean image.
  setMouseCallback("Result", resetSliderValues);
  
  cout << "Usage:" << endl 
  << "\tChange the weights using the sliders" << endl
  << "\tClick on the result window to reset sliders" << endl
  << "\tHit ESC to terminate program." << endl;

  waitKey(0);
  destroyAllWindows(); 
}

