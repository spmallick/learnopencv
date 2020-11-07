// Classification using SVM

#include "svm.h"
#include <ctype.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
using namespace std;

// create data
vector<vector<double>> generateData(int problemSize, int featureNum) {
  vector<vector<double>> data; 
  // our data 
  for(int i = 0; i < problemSize; i++) { 
    // create feature vector 
    vector<double> featureSet; 
    for(int j = 0; j < featureNum-1; j++) { 
      int value = 0; 
      int value_2 = 0; 
      // to make a gap between both classes 
      while(abs(value_2 - value) < 40){ 
        value = rand() % 1000; 
        value_2 = rand() % 1000; 
      } 
      featureSet.push_back(value); 
      featureSet.push_back(value_2); 
    } 
    data.push_back(featureSet); 
  } 
  return data; 
} 

// create labels 
vector<int> generateLabels(int labelsSize, vector< vector<double>> data) {
  // create labels vector 
  vector<int> labels; 
  for(int i = 0; i < labelsSize; i++) { 
    if(data[i][0] > data[i][1]) { 
      labels.push_back(1); 
    }
    else { 
      labels.push_back(-1); 
    } 
  } 
  // introduce noise in the data 
  for(int i = 0; i < labelsSize; i++) { 
    // invert label only for a few points 
    if(rand() % 1000 > 980) { 
      labels[i] = -1 * labels[i]; // invert the label 
    } 
  }
  return labels; 
}

// utility function to scale data
vector<vector<double>> scale_data(vector<vector<double>> data) {
  vector<int> minimum, maximum;
  vector<vector<double>> scaled_data;
  for(int i = 0; i < data.size(); i++) {
    vector<double> featureSet;
    for(int j = 0; j < data[i].size(); j++) {
      // scale data
      double value = 2 * (data[i][j] - minimum[j])/(maximum[j] - minimum[j]) -1;
      feature.push_back(value);
    }
    scaled_data.push_back(featureSet);
  }
  return scaled_data;
}

int main(){
	// Training and testing data
	int test_size = 300;
	int train_size = 700;
	int featureNum = 2;
	
	vector<vector<double>> test_data = generateData(test_size, featureNum);
	vector<int> test_labels = generateLabels(test_size, test_data);
	vector<vector<double>> train_data = generateData(train_size, featureNum);
	vector<int> train_labels = generateLabels(train_size, train_data);
	
	// Scale data
	train_data = scale_data(train_data);
	test_data = scale_data(test_data);
	
	// Train model on the dataset
	struct svm_parameter param; // parameters of svm
	struct svm_problem prob; // contains the training data in svm_node format
	// set parameters (optional)
	param.svm_type = C_SVC;
	param.kernel_type = RBF;

	// Number of training examples
	prob.l = train_size;

	// training dataset in svm_node matrix format
	svm_node** svm_x_train = (svm_node**)malloc((prob.l) * sizeof(svm_node*));

	// iterate over each sample
	for (int sample=0; sample < prob.l; sample++){
	  svm_node* x_space = (svm_node*)malloc((featureNum+1) * sizeof(svm_node));
	  for (int feature=0; feature < featureNum; feature++){
		// feature value
		x_space[feature].value= X_train[sample][feature];
		// feature index
		x_space[feature].index = feature+1;
	  }
	  // each sample's last feature should be -1 in libSVM
	  x_space[featureNum].index = -1;
	  svm_x_train[sample] = x_space;
	}

	// store training data in prob
	prob.x = svm_x_train;

	// store labels
	prob.y = (double *)malloc(prob.l * sizeof(double));
	for (int sample = 0; sample < prob.l; sample++){
	  prob.y[sample] = train_labels[sample];
	}

	// train the model
	struct svm_model *model;
	model = svm_train(&prob, &param);
	
	// Evaluating the trained model on test dataset
	// svm_predict returns the predicted value in C++
	int prediction;

	// iterate over each test sample
	for (int sample=0; sample < prob.l; sample++){
	  svm_node* x_space = (svm_node*)malloc((featureNum+1) * sizeof(svm_node));
	  for (int feature=0; feature < featureNum; feature++){
		// feature value
		x_space[feature].value= X_test[sample][feature];
		// feature index
		x_space[feature].index = feature+1;
	  }
	  // each sample's last feature should be -1 in libSVM
	  x_space[featureNum].index = -1;
	  prediction = svm_predict(model, x_space);
	  std::cout << "Prediction: " << prediction << ", Groundtruth: " << test_labels[sample] << std::endl;
	}
}