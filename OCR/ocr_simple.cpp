#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    string outText;
    string imPath = argv[1];

    // Create Tesseract object
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
  
    /*
     Initialize OCR engine to use English (eng) and The LSTM 
     OCR engine.
     
     
     There are four OCR Engine Mode (oem) available
     
     OEM_TESSERACT_ONLY             Legacy engine only.
     OEM_LSTM_ONLY                  Neural nets LSTM engine only.
     OEM_TESSERACT_LSTM_COMBINED    Legacy + LSTM engines.
     OEM_DEFAULT                    Default, based on what is available.
    */
  
    ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
  
  
    // Set Page segmentation mode to PSM_AUTO (3)
    // Other important psm modes will be discussed in a future post.
    ocr->SetPageSegMode(tesseract::PSM_AUTO);
  

    // Open input image using OpenCV
    Mat im = cv::imread(imPath, IMREAD_COLOR);
  
    // Set image data
    ocr->SetImage(im.data, im.cols, im.rows, 3, im.step);
    
    // Run Tesseract OCR on image
    outText = string(ocr->GetUTF8Text());

    // print recognized text
    cout << outText << endl;

    // Destroy used object and release memory
    ocr->End();
  
    return EXIT_SUCCESS;
}
