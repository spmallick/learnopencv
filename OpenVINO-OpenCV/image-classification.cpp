#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
 
using namespace std;
using namespace cv;
using namespace cv::dnn;

int main() {
  string caffe_root = "/home/ubuntu/caffe/";
  Mat image = imread("/home/ubuntu/caffe/examples/images/cat.jpg");
  string labels_file = "/home/ubuntu/caffe/data/ilsvrc12/synset_words.txt";
  string prototxt = "/home/ubuntu/caffe/models/bvlc_reference_caffenet/deploy.prototxt";
  string model = "/home/ubuntu/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";

  // load the labels file
  std::ifstream ifs(labels_file.c_str());
  if (!ifs.is_open())
    CV_Error(Error::StsError, "File " + labels_file + " not found");
    string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
  }
  blobFromImage(image, blob, 1, Size(224, 224), Scalar(104,117,123));
  cout << "[INFO] loading model..." << endl;
  Net net = readNetFromCaffe(prototxt, model);
  net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
  net.setPreferableTarget(DNN_TARGET_CPU);
  
  // set the blob as input to the network and perform a forward-pass to
  // obtain our output classification
  net.setInput(blob)
  preds = net.forward()
  
  double freq = getTickFrequency() / 1000;
  std::vector<double> layersTimes;
  double t = net.getPerfProfile(layersTimes) / freq;
  cout << "[INFO] classification took " << t << " ms" << endl;

  return 0;
}
