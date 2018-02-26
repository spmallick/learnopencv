#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;


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


int main(int argc, char **argv)
{
  // Directory containing images
  string dirName = "images/";
  
  // Read images in the directory
  vector<Mat> images; 
  readImages(dirName, images);
  
  // Size of images. All images should be the same size. 
  Size sz = images[0].size(); 
  Mat szMat = (Mat_<double>(3,1) << sz.height, sz.width, 3);
  
  // Create data matrix for PCA.
  Mat data = createDataMatrix(images);
  
  // Calculate PCA of the data matrix
  cout << "Calculating PCA ...";
  cout.flush(); 
  PCA pca(data, Mat(), PCA::DATA_AS_ROW);
  cout << " DONE"<< endl;

  // Copy mean vector  
  Mat meanVector = pca.mean;

  // Copy eigen vectors.
  Mat eigenVectors = pca.eigenvectors;

  // Write size, mean and eigenvectors to file
  string filename("pcaParams.yml");
  cout << "Writing size, mean and eigenVectors to "  << filename <<  " ... "; 
  cout.flush(); 
  FileStorage file = FileStorage(filename, FileStorage::WRITE);
  file << "mean" <<  meanVector;
  file << "eigenVectors" << eigenVectors;
  file << "size" << szMat;
  file.release();
  cout << "DONE" << endl; 
  
  
}

