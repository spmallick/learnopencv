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


Size sz;

// Weights for the different eigenvectors
int sliderValues[NUM_EIGEN_FACES];

// Matrices for average (mean) and eigenvectors
Mat averageFace;
vector<Mat> eigenFaces;

// Read jpg files from the directory
void readFileNames(string dirName, vector<string> &imageFnames)
{
  
  cout << "Reading images from " << dirName;
  
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
      string temp_name = ent->d_name;
      files.push_back(temp_name);
      
    }
    std::sort(files.begin(),files.end());
    for(int it=0;it<files.size();it++)
    {
      string path = dirName;
      string fname=files[it];
      
      
      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
    }
    closedir (dir);
  }
  
  cout << "... " << files.size() << " files read"<< endl;
  
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
  // numImages = number of images in the dataset.
  
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
    Mat eigenFace = eigenFaces[i];
    
    // OpenCV does not allow slider values to be negative.
    // So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
    double weight = sliderValues[i] - MAX_SLIDER_VALUE/2;
    output = output + eigenFace * weight;
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
  string dirName = "images";
  
  // Add slash to directory name if missing
  if (!dirName.empty() && dirName.back() != '/')
    dirName += '/';
  
  
  // Read images in the directory
  vector<string> imageNames;
  readFileNames(dirName, imageNames);
  
  
  // Exit program if no images are found
  if(imageNames.empty())exit(EXIT_FAILURE);
  
  // Read images
  vector <Mat> images;
  for(size_t i = 0; i < imageNames.size(); i++)
  {
    Mat img = imread(imageNames[i]);
    if(!img.data)
    {
      cout << "image " << imageNames[i] << " not read properly" << endl;
    }
    else
    {
      if ( i == 0)
      {
        // Set size of images based on the first image.
        // The rest of the code assumes all images are of this size.
        sz = img.size();
      }
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
  
  // Create data matrix for PCA.
  Mat data = createDataMatrix(images);
  
  // Calculate PCA of the data matrix
  cout << "Calculating PCA ...";
  PCA pca(data, Mat(), PCA::DATA_AS_ROW, 10);
  cout << " DONE"<< endl;
  
  // Extract mean vector
  averageFace = pca.mean;
  
  // Find eigen vectors.
  Mat eigenvectors = pca.eigenvectors;
  
  // Reshape mean vector to obtain average face
  averageFace = averageFace.reshape(3,sz.height);
  
  // Reshape Eigenvectors to obtain EigenFaces
  for(int i = 0; i < NUM_EIGEN_FACES; i++)
  {
      Mat eigenFace = eigenvectors.row(i).reshape(3,sz.height);
      eigenFaces.push_back(eigenFace);
  }
  
  
  // Show mean face image
  Mat display;
  resize(averageFace, display, Size(), 2, 2);
  
  namedWindow("Result", CV_WINDOW_AUTOSIZE);
  imshow("Result", display);
  
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

