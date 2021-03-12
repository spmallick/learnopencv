#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "guidedfilter.h"

std::pair<cv::Mat, cv::Mat> get_illumination_channel(cv::Mat I, float w) {
	int N = I.size[0];
	int M = I.size[1];
	cv::Mat darkch = cv::Mat::zeros(cv::Size(M, N), CV_32FC1);
	cv::Mat brightch = cv::Mat::zeros(cv::Size(M, N), CV_32FC1);

	//Padding for dark and bright channels on all the edges
	int padding = int(w/2);
	cv::Mat padded = cv::Mat::zeros(cv::Size(M + 2*padding, N + 2*padding), CV_32FC3);

	for (int i=padding; i < padding + M; i++) {
		for (int j=padding; j < padding + N; j++) {
			padded.at<cv::Vec3f>(j, i).val[0] = (float)I.at<cv::Vec3b>(j-padding, i-padding).val[0]/255;
			padded.at<cv::Vec3f>(j, i).val[1] = (float)I.at<cv::Vec3b>(j-padding, i-padding).val[1]/255;
			padded.at<cv::Vec3f>(j, i).val[2] = (float)I.at<cv::Vec3b>(j-padding, i-padding).val[2]/255;
		}
	}

	for (int i=0; i < darkch.size[1]; i++) {
		int col_up, row_up;
		
		col_up = int(i+w);

		for (int j=0; j < darkch.size[0]; j++) {
			double minVal, maxVal;

			row_up = int(j+w);

			//Get the min and max pixel values in each winow of size w
			cv::minMaxLoc(padded.colRange(i, col_up).rowRange(j, row_up), &minVal, &maxVal);

			//Dark channel obtained using minMaxLoc to get the lowest pixel value in that block
			darkch.at<float>(j,i) = minVal;

			//Bright channel obtained using minMaxLoc to get the highest pixel value in that block
			brightch.at<float>(j,i) = maxVal;
		}
	}

	return std::make_pair(darkch, brightch);
}

cv::Mat get_atmosphere(cv::Mat I, cv::Mat brightch, float p=0.1) {
	int N = brightch.size[0];
	int M = brightch.size[1];

	cv::Mat flatI(cv::Size(1, N*M), CV_8UC3);
	std::vector<std::pair<float, int>> flatBright;

	//Flattening the image I into flatI
	for (int i=0; i < M; i++) {
		for (int j=0; j < N; j++) {
			int index = i*N + j;
			flatI.at<cv::Vec3b>(index, 0).val[0] = I.at<cv::Vec3b>(j, i).val[0];
			flatI.at<cv::Vec3b>(index, 0).val[1] = I.at<cv::Vec3b>(j, i).val[1];
			flatI.at<cv::Vec3b>(index, 0).val[2] = I.at<cv::Vec3b>(j, i).val[2];

			//Storing the bright channels in flatBright vector along with index inorder to get the sorted values as well as indicies
			flatBright.push_back(std::make_pair(-brightch.at<float>(j, i), index));
		}
	}


	//Sorting according to maximum intensity and slicing the array to include only the top ten percent (p = 0.1) of pixels
	//To get descending order, added -ve sign to the flatBright values
	sort(flatBright.begin(), flatBright.end());

	cv::Mat A = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);

	for (int k=0; k < int(M*N*p); k++) {
		int sindex = flatBright[k].second;
		A.at<float>(0, 0) = A.at<float>(0, 0) + (float)flatI.at<cv::Vec3b>(sindex, 0).val[0];
		A.at<float>(1, 0) = A.at<float>(1, 0) + (float)flatI.at<cv::Vec3b>(sindex, 0).val[1];
		A.at<float>(2, 0) = A.at<float>(2, 0) + (float)flatI.at<cv::Vec3b>(sindex, 0).val[2];
	}

	A = A/int(M*N*p);

	return A/255;
}

cv::Mat get_initial_transmission(cv::Mat A, cv::Mat brightch) {
	double A_n, A_x, minVal, maxVal;
	cv::minMaxLoc(A, &A_n, &A_x);
	cv::Mat init_t(brightch.size(), CV_32FC1);
	init_t = brightch.clone();

	//Finding initial transmission map according to the above formula
	init_t = (init_t - A_x)/(1.0 - A_x);

	//Normalize initial transmission map
	cv::minMaxLoc(init_t, &minVal, &maxVal);
	init_t = (init_t - minVal)/(maxVal - minVal);

	return init_t;
}


cv::Mat reduce_init_t(cv::Mat init_t) {
	cv::Mat mod_init_t(init_t.size(), CV_8UC1);

	//The transmission map received was normalized so it is converted to pixels	having values between 0-255
	for (int i=0; i < init_t.size[1]; i++) {
		for (int j=0; j < init_t.size[0]; j++) {
			mod_init_t.at<uchar>(j, i) = std::min((int)(init_t.at<float>(j, i)*255), 255);
		}
	}

	int x[3] = {0, 32, 255};
	int f[3] = {0, 32, 48};

	cv::Mat table(cv::Size(1, 256), CV_8UC1);

	//Interpreting f according to x in range of k
	int l = 0;
	for (int k = 0; k < 256; k++) {
		if (k > x[l+1]) {
			l = l + 1;
		}

		float m  = (float)(f[l+1] - f[l])/(x[l+1] - x[l]);
		table.at<int>(k, 0) = (int)(f[l] + m*(k - x[l]));
	}

	//Lookup table
	cv::LUT(mod_init_t, table, mod_init_t);

	//The transmission map is normalized before returning it
	for (int i=0; i < init_t.size[1]; i++) {
		for (int j=0; j < init_t.size[0]; j++) {
			init_t.at<float>(j, i) = (float)mod_init_t.at<uchar>(j, i)/255;
		}
	}

	return init_t;
}

cv::Mat get_corrected_transmission(cv::Mat I, cv::Mat A, cv::Mat darkch, cv::Mat brightch, cv::Mat init_t, float alpha, float omega, int w) {
	cv::Mat im3(I.size(), CV_32FC3);

	for (int i=0; i < I.size[1]; i++) {
		for (int j=0; j < I.size[0]; j++) {
			im3.at<cv::Vec3f>(j, i).val[0] = (float)I.at<cv::Vec3b>(j, i).val[0]/A.at<float>(0, 0);
			im3.at<cv::Vec3f>(j, i).val[1] = (float)I.at<cv::Vec3b>(j, i).val[1]/A.at<float>(1, 0);
			im3.at<cv::Vec3f>(j, i).val[2] = (float)I.at<cv::Vec3b>(j, i).val[2]/A.at<float>(2, 0);
		}
	}

	cv::Mat dark_c, dark_t, diffch;

	//Getting dark channel transmission map
	std::pair<cv::Mat, cv::Mat> illuminate_channels = get_illumination_channel(im3, w);
	dark_c = illuminate_channels.first;

	//Finding dark transmission map using omega (0.75) which will be used to correct the initial transmission map.
	dark_t = 1 - omega*dark_c;

	//Initializing corrected transmission map with initial transmission map as its values will remain
	//the same as the initial transmission map when the difference between them is less than alpha
	cv::Mat corrected_t = init_t;

	//Finding difference between transmission maps
	diffch = brightch - darkch;

	for (int i=0; i < diffch.size[1]; i++) {
		for (int j=0; j < diffch.size[0]; j++) {

			//if difference between the transmission greater than alpha (0.4) the transmission map is corrected by
			//taking their product
			if (diffch.at<float>(j, i) < alpha) {
				corrected_t.at<float>(j, i) = abs(dark_t.at<float>(j, i)*init_t.at<float>(j, i)); 
			}
		}
	}

	return corrected_t;
}

cv::Mat get_final_image(cv::Mat I, cv::Mat A, cv::Mat refined_t, float tmin) {
	cv::Mat J(I.size(), CV_32FC3);

	for (int i=0; i < refined_t.size[1]; i++) {
		for (int j=0; j < refined_t.size[0]; j++) {
			//Value of refined_t (2D refined map) at (j, i) is considered if it is >= tmin. 
			float temp = refined_t.at<float>(j, i);

			if (temp < tmin) {
				temp = tmin;
			}

			//Finding result using the formula given at top
			J.at<cv::Vec3f>(j, i).val[0] = (I.at<cv::Vec3f>(j, i).val[0] - A.at<float>(0,0))/temp + A.at<float>(0,0);
			J.at<cv::Vec3f>(j, i).val[1] = (I.at<cv::Vec3f>(j, i).val[1] - A.at<float>(1,0))/temp + A.at<float>(1,0);
			J.at<cv::Vec3f>(j, i).val[2] = (I.at<cv::Vec3f>(j, i).val[2] - A.at<float>(2,0))/temp + A.at<float>(2,0);
		}
	}

	double minVal, maxVal;
	cv::minMaxLoc(J, &minVal, &maxVal);

	//Normalize the image J
	for (int i=0; i < J.size[1]; i++) {
		for (int j=0; j < J.size[0]; j++) {
			J.at<cv::Vec3f>(j, i).val[0] = (J.at<cv::Vec3f>(j, i).val[0] - minVal)/(maxVal - minVal);
			J.at<cv::Vec3f>(j, i).val[1] = (J.at<cv::Vec3f>(j, i).val[1] - minVal)/(maxVal - minVal);
			J.at<cv::Vec3f>(j, i).val[2] = (J.at<cv::Vec3f>(j, i).val[2] - minVal)/(maxVal - minVal);
		}
	}

	return J;
}

cv::Mat dehaze(cv::Mat img, float tmin=0.1, int w = 15, float alpha=0.4, float omega=0.75, float p=0.1, double eps=1e-3, bool reduce=false) {
	std::pair<cv::Mat, cv::Mat> illuminate_channels = get_illumination_channel(img, w);
	cv::Mat Idark = illuminate_channels.first;
	cv::Mat Ibright = illuminate_channels.second;

	cv::Mat A = get_atmosphere(img, Ibright);

	cv::Mat init_t = get_initial_transmission(A, Ibright);

	if (reduce) {
		init_t = reduce_init_t(init_t);
	}

	cv::Mat corrected_t = get_corrected_transmission(img, A, Idark, Ibright, init_t, alpha, omega, w); 

	cv::Mat I(img.size(), CV_32FC3), normI;

	for (int i=0; i < img.size[1]; i++) {
		for (int j=0; j < img.size[0]; j++) {
			I.at<cv::Vec3f>(j, i).val[0] = (float)img.at<cv::Vec3b>(j, i).val[0]/255;
			I.at<cv::Vec3f>(j, i).val[1] = (float)img.at<cv::Vec3b>(j, i).val[1]/255;
			I.at<cv::Vec3f>(j, i).val[2] = (float)img.at<cv::Vec3b>(j, i).val[2]/255;
		}
	}

	double minVal, maxVal;
	cv::minMaxLoc(I, &minVal, &maxVal);
	normI = (I - minVal)/(maxVal - minVal);

	//Applying guided filter
	cv::Mat refined_t(normI.size(), CV_32FC1);
	refined_t = guidedFilter(normI, corrected_t, w, eps);

	cv::Mat J_refined = get_final_image(I, A, refined_t, tmin);

	cv::Mat enhanced(img.size(), CV_8UC3);

	for (int i=0; i < img.size[1]; i++) {
		for (int j=0; j < img.size[0]; j++) {
			enhanced.at<cv::Vec3b>(j, i).val[0] = std::min((int)(J_refined.at<cv::Vec3f>(j, i).val[0]*255), 255);
			enhanced.at<cv::Vec3b>(j, i).val[1] = std::min((int)(J_refined.at<cv::Vec3f>(j, i).val[1]*255), 255);
			enhanced.at<cv::Vec3b>(j, i).val[2] = std::min((int)(J_refined.at<cv::Vec3f>(j, i).val[2]*255), 255);
		}
	}

	cv::Mat f_enhanced;
	cv::detailEnhance(enhanced, f_enhanced, 10, 0.15);
	cv::edgePreservingFilter(f_enhanced, f_enhanced, 1, 64, 0.2);

	return f_enhanced;
}

int main() {
	cv::Mat img = cv::imread("dark.png");
	cv::Mat out_img = dehaze(img);
	cv::Mat out_img2 = dehaze(img,0.1,15,0.4,0.75,0.1,1e-3,true);
	cv::imshow("original",img);
	cv::imshow("F_enhanced", out_img);
	cv::imshow("F_enhanced2", out_img2);
	cv::waitKey(0);
	return 0;
}
