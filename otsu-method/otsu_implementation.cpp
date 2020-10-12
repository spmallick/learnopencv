#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(){
	
	// read the image in BGR format
	Mat testImage = imread("boat.jpg", 0);
	int bins_num = 256;
	
	// Get the histogram
	long double histogram[256];
 
    // initialize all intensity values to 0
    for(int i = 0; i < 256; i++)
    {
        histogram[i] = 0;
    }
 
    // calculate the no of pixels for each intensity values
    for(int y = 0; y < testImage.rows; y++)
        for(int x = 0; x < testImage.cols; x++)
            histogram[(int)testImage.at<uchar>(y,x)]++;
		
	// Calculate the bin_edges
	long double bin_edges[256];
	bin_edges[0] = 0.0;
	long double increment = 0.99609375;
	for(int i = 1; i < 256; i++){
		bin_edges[i] = bin_edges[i-1] + increment;
	}
	
	// Calculate bin_mids
	long double bin_mids[256];
	for(int i = 0; i < 256; i++){
		bin_mids[i] = (bin_edges[i] + bin_edges[(uchar)(i+1)])/2;
	}
	
	// Calculate weight 1 and weight 2
	long double weight1[256];
	weight1[0] = histogram[0];
	for(int i = 1; i < 256; i++){
		weight1[i] = histogram[i] + weight1[i-1];
	}
	int total_sum=0;
	for(int i = 0; i < 256; i++){
		total_sum = total_sum + histogram[i];
	}
	long double weight2[256];
	weight2[0] = total_sum;
	for(int i = 1; i < 256; i++){
		weight2[i] = weight2[i-1] - histogram[i - 1];
	}
	
	// Calculate mean 1 and mean 2
	long double histogram_bin_mids[256];
	for(int i = 0; i < 256; i++){
		histogram_bin_mids[i] = histogram[i] * bin_mids[i];
	}
	long double cumsum_mean1[256];
	cumsum_mean1[0] = histogram_bin_mids[0];
	for(int i = 1; i < 256; i++){
		cumsum_mean1[i] = cumsum_mean1[i-1] + histogram_bin_mids[i];
	}
	long double cumsum_mean2[256];
	cumsum_mean2[0] = histogram_bin_mids[255];
	for(int i = 1, j=254; i < 256 && j>=0; i++, j--){
		cumsum_mean2[i] = cumsum_mean2[i-1] + histogram_bin_mids[j];
	}
	long double mean1[256];
	for(int i = 0; i < 256; i++){
		mean1[i] = cumsum_mean1[i] / weight1[i];
	}
	long double mean2[256];
	for(int i = 0, j = 255; i < 256 && j >= 0; i++, j--){
		mean2[j] = cumsum_mean2[i] / weight2[j];
	}


	// Calculate Inter_class_variance
	long double Inter_class_variance[255];
	long double dnum = 10000000000;
	for(int i = 0; i < 255; i++){
		Inter_class_variance[i] = ((weight1[i] * weight2[i] * (mean1[i] - mean2[i+1])) / dnum) * (mean1[i] - mean2[i+1]);
	}

	// Get the maximum value
	long double maxi = 0;
	int getmax = 0;
	for(int i = 0;i < 255; i++){
		if(maxi < Inter_class_variance[i]){
			maxi = Inter_class_variance[i];
			getmax = i;
		}
	}
	
	cout << "Otsu's algorithm implementation thresholding result: " << bin_mids[getmax];

	
	return 0;
}
