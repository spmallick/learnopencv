#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>
using namespace cv;
using namespace std;
// Functions
static void read_imgList(const string& filename, vector<Mat>& images) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line;
    while (getline(file, line)) {
        images.push_back(imread(line, 0));
    }
}
void readFileNames(string dirName, vector<string> &imageFnames)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;
  
  //image extensions
  string imgExt = "jpg";
  vector<string> files;
  int c = 0;
  if ((dir = opendir (dirName.c_str())) != NULL)
  {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL && c< 25)
    {
      if(strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name,"..") == 0 )
      {
        continue;
      }
      string temp_name = ent->d_name;
      files.push_back(temp_name);
      c++;
      
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
static  Mat formatImagesForPCA(const vector<Mat> &data)
{
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_32F);
    }
    return dst;
}
static Mat toGrayscale(InputArray _src) {
    Mat src = _src.getMat();
    // only allow one channel
    if(src.channels() != 1) {
        CV_Error(Error::StsBadArg, "Only Matrices with one channel are supported");
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}
struct params
{
    Mat data;
    int ch;
    int rows;
    PCA pca;
    string winName;
};
static void onTrackbar(int pos, void* ptr)
{
    cout << "Retained Variance = " << pos << "%   ";
    cout << "re-calculating PCA..." << std::flush;
    double var = pos / 100.0;
    struct params *p = (struct params *)ptr;
    p->pca = PCA(p->data, cv::Mat(), PCA::DATA_AS_ROW, var);

    Mat point = p->pca.project(p->data.row(0));
    Mat reconstruction = p->pca.backProject(point);
    reconstruction = reconstruction.reshape(p->ch, p->rows);
    reconstruction = toGrayscale(reconstruction);
    //cout << p->pca.eigenvectors<< endl;
    //imshow("Eigen vector",p->pca.eigenvectors);
    imshow(p->winName, reconstruction);
    cout << "done!   # of principal components: " << p->pca.eigenvectors.rows << endl;
}
// Main
int main(int argc, char** argv)
{
    /*cv::CommandLineParser parser(argc, argv, "{@input||image list}{help h||show help message}");
    if (parser.has("help"))
    {
        parser.printMessage();
        exit(0);
    }
    // Get the path to your CSV.
    string imgList = parser.get<string>("@input");
    if (imgList.empty())
    {
        parser.printMessage();
        exit(1);
    }
    // vector to hold the images
    vector<Mat> images;
    // Read in the data. This can fail if not valid
    try {
        read_imgList(imgList, images);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << imgList << "\". Reason: " << e.msg << endl;
        exit(1);
    }*/
    // Quit if there are not enough images for this demo.
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
        Mat img = imread(imageNames[i],0);
        if(!img.data)
        {
          cout << "image " << imageNames[i] << " not read properly" << endl;
        }
        else
        {
            img.convertTo(img, CV_32FC1, 1/255.0);
            resize(img, img, Size(200,200),0,0, INTER_CUBIC);
            images.push_back(img);
        }
    }
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }
    // Reshape and stack images into a rowMatrix
    Mat average= Mat::zeros(200,200, CV_32FC1);
    for(size_t i=0; i<images.size();i++){
        average = average+images[i];
    }
    average = average/(images.size());
    for(size_t i=0; i< images.size();i++)
    {
        images[i] = images[i]-average;
        //images[i] = images[i].reshape(1, 1);
        //imshow("reshape", images[0]);
    }
    Mat data = formatImagesForPCA(images);
    // perform PCA
    PCA pca(data, cv::Mat(), PCA::DATA_AS_ROW, 0.95); // trackbar is initially set here, also this is a common value for retainedVariance
    // Demonstration of the effect of retainedVariance on the first image
    Mat point = pca.project(data.row(0)); // project into the eigenspace, thus the image becomes a "point"
    Mat reconstruction = pca.backProject(point); // re-create the image from the "point"
    reconstruction = reconstruction.reshape(images[0].channels(), images[0].rows); // reshape from a row vector into image shape
    reconstruction = toGrayscale(reconstruction); // re-scale for displaying purposes
    // init highgui window
    string winName = "Reconstruction | press 'q' to quit";
    namedWindow(winName, WINDOW_NORMAL);
    // params struct to pass to the trackbar handler
    params p;
    p.data = data;
    p.ch = images[0].channels();
    p.rows = images[0].rows;
    p.pca = pca;
    p.winName = winName;
    // create the tracbar
    int pos = 95;
    createTrackbar("Retained Variance (%)", winName, &pos, 100, onTrackbar, (void*)&p);
    // display until user presses q
    imshow(winName, reconstruction);
    int key = 0;
    while(key != 'q')
        key = waitKey();
   return 0;
}