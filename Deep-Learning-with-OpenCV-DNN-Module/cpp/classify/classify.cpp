#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int main(int, char**) {
    std::vector<std::string> class_names;
    ifstream ifs(string("../../input/classification_classes_ILSVRC2012.txt").c_str());
    string line;
    while (getline(ifs, line))
    {
        class_names.push_back(line);
    }  
    
    // load the neural network model
    auto model = readNet("../../input/DenseNet_121.prototxt", 
                        "../../input/DenseNet_121.caffemodel", 
                        "Caffe");
    
    // load the image from disk
    Mat image = imread("../../input/image_1.jpg");
    // create blob from image
    Mat blob = blobFromImage(image, 0.01, Size(224, 224), Scalar(104, 117, 123));

    // set the input blob for the neural network
    model.setInput(blob);
    // forward pass the image blob through the model
    Mat outputs = model.forward();

    Point classIdPoint;
    double final_prob;
    minMaxLoc(outputs.reshape(1, 1), 0, &final_prob, 0, &classIdPoint);
    int label_id = classIdPoint.x;

    // Print predicted class.
    string out_text = format("%s, %.3f", (class_names[label_id].c_str()), final_prob);
    // put the class name text on top of the image
    putText(image, out_text, Point(25, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0),
                2);
        
    imshow("Image", image);
    imwrite("../../outputs/result_image.jpg", image);
}
