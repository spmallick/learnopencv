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


// Weights for the different eigenvectors
int wt_0 = 0;
int wt_1 = 0;
int wt_2 = 0;
int wt_3 = 0;
int wt_4 = 0;
int wt_5 = 0;
int wt_6 = 0;
int wt_7 = 0;
int wt_8 = 0;
int wt_9 = 0;

int max_wt =255;

// Matrix for average and eigenvectors
Mat average, eigenvectors;
// Read jpg files from the directory
void readFileNames(string dirName, vector<string> &imageFnames)
{
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
  
}

// Format images according to the requirement of pca function
static  Mat formatImagesForPCA(const vector<Mat> &data)
{
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols*3, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_32F);
    }
    return dst;
}

// Add weighted eigen vectors to the average image
void calcFinalImage(int ,void *){
	Mat temp = average.clone();
	Mat eig1 = eigenvectors.row(0).reshape(3,200);
	Mat eig2 = eigenvectors.row(1).reshape(3,200);
	Mat eig3 = eigenvectors.row(2).reshape(3,200);
	Mat eig4 = eigenvectors.row(3).reshape(3,200);
	Mat eig5 = eigenvectors.row(4).reshape(3,200);
	Mat eig6 = eigenvectors.row(5).reshape(3,200);
	Mat eig7 = eigenvectors.row(6).reshape(3,200);
	Mat eig8 = eigenvectors.row(7).reshape(3,200);
	Mat eig9 = eigenvectors.row(8).reshape(3,200);
	Mat eig10 = eigenvectors.row(9).reshape(3,200);
	temp = temp + eig1*wt_0;
	temp = temp + eig2*wt_1;
	temp = temp + eig3*wt_2;
	temp = temp + eig4*wt_3;
	temp = temp + eig5*wt_4;
	temp = temp + eig6*wt_5;
	temp = temp + eig7*wt_6;
	temp = temp + eig8*wt_7;
	temp = temp + eig9*wt_8;
	temp = temp + eig10*wt_9;
	imshow("Result", temp);

}

int main(int argc, char **argv){
	cout << "Hello"<<endl;
	string dirName = "../PCA/images";
	// Add slash to directory name if missing
  	if (!dirName.empty() && dirName.back() != '/')
    	dirName += '/';
    // Read images in the directory
  	vector<string> imageNames;
  	readFileNames(dirName, imageNames);

  	// Exit program if no images are found or if the number of image files does not match with the number of point files
  	if(imageNames.empty())exit(EXIT_FAILURE);

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
	        img.convertTo(img, CV_32FC3, 1/255.0);
	        resize(img, img, Size(200,200),0,0, INTER_CUBIC);
	        images.push_back(img);
	    }
  	}

    // Initialize average image
  	average= Mat::zeros(200,200, CV_32FC3);
  	for(size_t i=0; i<images.size();i++){
  		average = average+images[i];
  	}
    // Calculate average image
  	average = average/(images.size());

  	for(size_t i=0; i< images.size();i++)
  		images[i] = images[i]-average;

  	cout << "Formatting images..."<<endl;
  	Mat data = formatImagesForPCA(images);
  	cout << "DONE" << endl;

  	cout << "Calculating PCA..." << endl;
    // Calculate PCA of the data matrix
  	PCA pca(data, cv::Mat(), PCA::DATA_AS_ROW, 10);
  	cout << "DONE"<< endl;
  	eigenvectors = pca.eigenvectors; 

  	namedWindow("Trackbars", CV_WINDOW_AUTOSIZE);
    // Show average image
  	imshow("Average", average);

    // Create trackbars
  	createTrackbar( "Weight0", "Trackbars", &wt_0, max_wt, calcFinalImage);
  	createTrackbar( "Weight1", "Trackbars", &wt_1, max_wt, calcFinalImage);
  	createTrackbar( "Weight2", "Trackbars", &wt_2, max_wt, calcFinalImage);
  	createTrackbar( "Weight3", "Trackbars", &wt_3, max_wt, calcFinalImage);
  	createTrackbar( "Weight4", "Trackbars", &wt_4, max_wt, calcFinalImage);
  	createTrackbar( "Weight5", "Trackbars", &wt_5, max_wt, calcFinalImage);
  	createTrackbar( "Weight6", "Trackbars", &wt_6, max_wt, calcFinalImage);
  	createTrackbar( "Weight7", "Trackbars", &wt_7, max_wt, calcFinalImage);
  	createTrackbar( "Weight8", "Trackbars", &wt_8, max_wt, calcFinalImage);
  	createTrackbar( "Weight9", "Trackbars", &wt_9, max_wt, calcFinalImage);
  	waitKey(0);
}

