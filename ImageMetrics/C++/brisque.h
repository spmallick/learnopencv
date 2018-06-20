#ifndef JD_BRISQUE
#define JD_BRISQUE
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <string.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
//#define M_PI 3.14159265358979323846

extern float rescale_vector[36][2];

template<class T> class Image
{
    private:

        Mat imgp;
    public:
        Image(Mat img=0)
        {
            imgp=img.clone();
        }
        ~Image()
        {
            imgp=0;
        }
        Mat equate(Mat img) 
        {
            img = imgp.clone();
            return img;
        }
        void showimage() {
            imshow("imgp", imgp);
            waitKey(0);
            destroyAllWindows();
        }
        inline T* operator[](const int rowIndx)
        {
            //imgp->data and imgp->width
            return (T*)(imgp.data + rowIndx*imgp.step);
        }
};

typedef Image<double> BwImage;
//function declarations
Mat AGGDfit(Mat structdis, double& lsigma_best, double& rsigma_best, double& gamma_best);
void ComputeBrisqueFeature(Mat& orig, vector<double>& featurevector);
void trainModel();
float computescore(string imname);

    template <typename Type>
void  printVector(vector<Type> vec)
{
    for(int i=0; i<vec.size(); i++)
    {
        cout<<i+1<<":"<<vec[i]<<endl;
    }
}

    template <typename Type>
void printVectortoFile(const char* filename , vector<Type> vec,float score)
{
    FILE* fid = fopen(filename,"a");
    //cout<<"file opened"<<endl;
    fprintf(fid,"%f ",score);
    for(int itr_param = 0; itr_param < vec.size();itr_param++)
        fprintf(fid,"%d:%f ",itr_param+1,vec[itr_param]);
    fprintf(fid,"\n");
    fclose(fid);
}


#endif
