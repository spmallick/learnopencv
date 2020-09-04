//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.objdetect;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Size;
import org.opencv.utils.Converters;

// C++: class HOGDescriptor
/**
 * Implementation of HOG (Histogram of Oriented Gradients) descriptor and object detector.
 *
 * the HOG descriptor algorithm introduced by Navneet Dalal and Bill Triggs CITE: Dalal2005 .
 *
 * useful links:
 *
 * https://hal.inria.fr/inria-00548512/document/
 *
 * https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
 *
 * https://software.intel.com/en-us/ipp-dev-reference-histogram-of-oriented-gradients-hog-descriptor
 *
 * http://www.learnopencv.com/histogram-of-oriented-gradients
 *
 * http://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial
 */
public class HOGDescriptor {

    protected final long nativeObj;
    protected HOGDescriptor(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static HOGDescriptor __fromPtr__(long addr) { return new HOGDescriptor(addr); }

    // C++: enum DescriptorStorageFormat
    public static final int
            DESCR_FORMAT_COL_BY_COL = 0,
            DESCR_FORMAT_ROW_BY_ROW = 1;


    // C++: enum <unnamed>
    public static final int
            DEFAULT_NLEVELS = 64;


    // C++: enum HistogramNormType
    public static final int
            L2Hys = 0;


    //
    // C++:   cv::HOGDescriptor::HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture = 1, double _winSigma = -1, HOGDescriptor_HistogramNormType _histogramNormType = HOGDescriptor::L2Hys, double _L2HysThreshold = 0.2, bool _gammaCorrection = false, int _nlevels = HOGDescriptor::DEFAULT_NLEVELS, bool _signedGradient = false)
    //

    /**
     *
     *     @param _winSize sets winSize with given value.
     *     @param _blockSize sets blockSize with given value.
     *     @param _blockStride sets blockStride with given value.
     *     @param _cellSize sets cellSize with given value.
     *     @param _nbins sets nbins with given value.
     *     @param _derivAperture sets derivAperture with given value.
     *     @param _winSigma sets winSigma with given value.
     *     @param _histogramNormType sets histogramNormType with given value.
     *     @param _L2HysThreshold sets L2HysThreshold with given value.
     *     @param _gammaCorrection sets gammaCorrection with given value.
     *     @param _nlevels sets nlevels with given value.
     *     @param _signedGradient sets signedGradient with given value.
     */
    public HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold, boolean _gammaCorrection, int _nlevels, boolean _signedGradient) {
        nativeObj = HOGDescriptor_0(_winSize.width, _winSize.height, _blockSize.width, _blockSize.height, _blockStride.width, _blockStride.height, _cellSize.width, _cellSize.height, _nbins, _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold, _gammaCorrection, _nlevels, _signedGradient);
    }

    /**
     *
     *     @param _winSize sets winSize with given value.
     *     @param _blockSize sets blockSize with given value.
     *     @param _blockStride sets blockStride with given value.
     *     @param _cellSize sets cellSize with given value.
     *     @param _nbins sets nbins with given value.
     *     @param _derivAperture sets derivAperture with given value.
     *     @param _winSigma sets winSigma with given value.
     *     @param _histogramNormType sets histogramNormType with given value.
     *     @param _L2HysThreshold sets L2HysThreshold with given value.
     *     @param _gammaCorrection sets gammaCorrection with given value.
     *     @param _nlevels sets nlevels with given value.
     */
    public HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold, boolean _gammaCorrection, int _nlevels) {
        nativeObj = HOGDescriptor_1(_winSize.width, _winSize.height, _blockSize.width, _blockSize.height, _blockStride.width, _blockStride.height, _cellSize.width, _cellSize.height, _nbins, _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold, _gammaCorrection, _nlevels);
    }

    /**
     *
     *     @param _winSize sets winSize with given value.
     *     @param _blockSize sets blockSize with given value.
     *     @param _blockStride sets blockStride with given value.
     *     @param _cellSize sets cellSize with given value.
     *     @param _nbins sets nbins with given value.
     *     @param _derivAperture sets derivAperture with given value.
     *     @param _winSigma sets winSigma with given value.
     *     @param _histogramNormType sets histogramNormType with given value.
     *     @param _L2HysThreshold sets L2HysThreshold with given value.
     *     @param _gammaCorrection sets gammaCorrection with given value.
     */
    public HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold, boolean _gammaCorrection) {
        nativeObj = HOGDescriptor_2(_winSize.width, _winSize.height, _blockSize.width, _blockSize.height, _blockStride.width, _blockStride.height, _cellSize.width, _cellSize.height, _nbins, _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold, _gammaCorrection);
    }

    /**
     *
     *     @param _winSize sets winSize with given value.
     *     @param _blockSize sets blockSize with given value.
     *     @param _blockStride sets blockStride with given value.
     *     @param _cellSize sets cellSize with given value.
     *     @param _nbins sets nbins with given value.
     *     @param _derivAperture sets derivAperture with given value.
     *     @param _winSigma sets winSigma with given value.
     *     @param _histogramNormType sets histogramNormType with given value.
     *     @param _L2HysThreshold sets L2HysThreshold with given value.
     */
    public HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold) {
        nativeObj = HOGDescriptor_3(_winSize.width, _winSize.height, _blockSize.width, _blockSize.height, _blockStride.width, _blockStride.height, _cellSize.width, _cellSize.height, _nbins, _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold);
    }

    /**
     *
     *     @param _winSize sets winSize with given value.
     *     @param _blockSize sets blockSize with given value.
     *     @param _blockStride sets blockStride with given value.
     *     @param _cellSize sets cellSize with given value.
     *     @param _nbins sets nbins with given value.
     *     @param _derivAperture sets derivAperture with given value.
     *     @param _winSigma sets winSigma with given value.
     *     @param _histogramNormType sets histogramNormType with given value.
     */
    public HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType) {
        nativeObj = HOGDescriptor_4(_winSize.width, _winSize.height, _blockSize.width, _blockSize.height, _blockStride.width, _blockStride.height, _cellSize.width, _cellSize.height, _nbins, _derivAperture, _winSigma, _histogramNormType);
    }

    /**
     *
     *     @param _winSize sets winSize with given value.
     *     @param _blockSize sets blockSize with given value.
     *     @param _blockStride sets blockStride with given value.
     *     @param _cellSize sets cellSize with given value.
     *     @param _nbins sets nbins with given value.
     *     @param _derivAperture sets derivAperture with given value.
     *     @param _winSigma sets winSigma with given value.
     */
    public HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture, double _winSigma) {
        nativeObj = HOGDescriptor_5(_winSize.width, _winSize.height, _blockSize.width, _blockSize.height, _blockStride.width, _blockStride.height, _cellSize.width, _cellSize.height, _nbins, _derivAperture, _winSigma);
    }

    /**
     *
     *     @param _winSize sets winSize with given value.
     *     @param _blockSize sets blockSize with given value.
     *     @param _blockStride sets blockStride with given value.
     *     @param _cellSize sets cellSize with given value.
     *     @param _nbins sets nbins with given value.
     *     @param _derivAperture sets derivAperture with given value.
     */
    public HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture) {
        nativeObj = HOGDescriptor_6(_winSize.width, _winSize.height, _blockSize.width, _blockSize.height, _blockStride.width, _blockStride.height, _cellSize.width, _cellSize.height, _nbins, _derivAperture);
    }

    /**
     *
     *     @param _winSize sets winSize with given value.
     *     @param _blockSize sets blockSize with given value.
     *     @param _blockStride sets blockStride with given value.
     *     @param _cellSize sets cellSize with given value.
     *     @param _nbins sets nbins with given value.
     */
    public HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins) {
        nativeObj = HOGDescriptor_7(_winSize.width, _winSize.height, _blockSize.width, _blockSize.height, _blockStride.width, _blockStride.height, _cellSize.width, _cellSize.height, _nbins);
    }


    //
    // C++:   cv::HOGDescriptor::HOGDescriptor(String filename)
    //

    /**
     *
     *     @param filename The file name containing HOGDescriptor properties and coefficients for the linear SVM classifier.
     */
    public HOGDescriptor(String filename) {
        nativeObj = HOGDescriptor_8(filename);
    }


    //
    // C++:   cv::HOGDescriptor::HOGDescriptor()
    //

    /**
     * Creates the HOG descriptor and detector with default params.
     *
     *     aqual to HOGDescriptor(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9 )
     */
    public HOGDescriptor() {
        nativeObj = HOGDescriptor_9();
    }


    //
    // C++:  bool cv::HOGDescriptor::checkDetectorSize()
    //

    /**
     * Checks if detector size equal to descriptor size.
     * @return automatically generated
     */
    public boolean checkDetectorSize() {
        return checkDetectorSize_0(nativeObj);
    }


    //
    // C++:  bool cv::HOGDescriptor::load(String filename, String objname = String())
    //

    /**
     * loads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file.
     *     @param filename Path of the file to read.
     *     @param objname The optional name of the node to read (if empty, the first top-level node will be used).
     * @return automatically generated
     */
    public boolean load(String filename, String objname) {
        return load_0(nativeObj, filename, objname);
    }

    /**
     * loads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file.
     *     @param filename Path of the file to read.
     * @return automatically generated
     */
    public boolean load(String filename) {
        return load_1(nativeObj, filename);
    }


    //
    // C++:  double cv::HOGDescriptor::getWinSigma()
    //

    /**
     * Returns winSigma value
     * @return automatically generated
     */
    public double getWinSigma() {
        return getWinSigma_0(nativeObj);
    }


    //
    // C++:  size_t cv::HOGDescriptor::getDescriptorSize()
    //

    /**
     * Returns the number of coefficients required for the classification.
     * @return automatically generated
     */
    public long getDescriptorSize() {
        return getDescriptorSize_0(nativeObj);
    }


    //
    // C++: static vector_float cv::HOGDescriptor::getDaimlerPeopleDetector()
    //

    /**
     * Returns coefficients of the classifier trained for people detection (for 48x96 windows).
     * @return automatically generated
     */
    public static MatOfFloat getDaimlerPeopleDetector() {
        return MatOfFloat.fromNativeAddr(getDaimlerPeopleDetector_0());
    }


    //
    // C++: static vector_float cv::HOGDescriptor::getDefaultPeopleDetector()
    //

    /**
     * Returns coefficients of the classifier trained for people detection (for 64x128 windows).
     * @return automatically generated
     */
    public static MatOfFloat getDefaultPeopleDetector() {
        return MatOfFloat.fromNativeAddr(getDefaultPeopleDetector_0());
    }


    //
    // C++:  void cv::HOGDescriptor::compute(Mat img, vector_float& descriptors, Size winStride = Size(), Size padding = Size(), vector_Point locations = std::vector<Point>())
    //

    /**
     * Computes HOG descriptors of given image.
     *     @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
     *     @param descriptors Matrix of the type CV_32F
     *     @param winStride Window stride. It must be a multiple of block stride.
     *     @param padding Padding
     *     @param locations Vector of Point
     */
    public void compute(Mat img, MatOfFloat descriptors, Size winStride, Size padding, MatOfPoint locations) {
        Mat descriptors_mat = descriptors;
        Mat locations_mat = locations;
        compute_0(nativeObj, img.nativeObj, descriptors_mat.nativeObj, winStride.width, winStride.height, padding.width, padding.height, locations_mat.nativeObj);
    }

    /**
     * Computes HOG descriptors of given image.
     *     @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
     *     @param descriptors Matrix of the type CV_32F
     *     @param winStride Window stride. It must be a multiple of block stride.
     *     @param padding Padding
     */
    public void compute(Mat img, MatOfFloat descriptors, Size winStride, Size padding) {
        Mat descriptors_mat = descriptors;
        compute_1(nativeObj, img.nativeObj, descriptors_mat.nativeObj, winStride.width, winStride.height, padding.width, padding.height);
    }

    /**
     * Computes HOG descriptors of given image.
     *     @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
     *     @param descriptors Matrix of the type CV_32F
     *     @param winStride Window stride. It must be a multiple of block stride.
     */
    public void compute(Mat img, MatOfFloat descriptors, Size winStride) {
        Mat descriptors_mat = descriptors;
        compute_2(nativeObj, img.nativeObj, descriptors_mat.nativeObj, winStride.width, winStride.height);
    }

    /**
     * Computes HOG descriptors of given image.
     *     @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
     *     @param descriptors Matrix of the type CV_32F
     */
    public void compute(Mat img, MatOfFloat descriptors) {
        Mat descriptors_mat = descriptors;
        compute_3(nativeObj, img.nativeObj, descriptors_mat.nativeObj);
    }


    //
    // C++:  void cv::HOGDescriptor::computeGradient(Mat img, Mat& grad, Mat& angleOfs, Size paddingTL = Size(), Size paddingBR = Size())
    //

    /**
     *  Computes gradients and quantized gradient orientations.
     *     @param img Matrix contains the image to be computed
     *     @param grad Matrix of type CV_32FC2 contains computed gradients
     *     @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
     *     @param paddingTL Padding from top-left
     *     @param paddingBR Padding from bottom-right
     */
    public void computeGradient(Mat img, Mat grad, Mat angleOfs, Size paddingTL, Size paddingBR) {
        computeGradient_0(nativeObj, img.nativeObj, grad.nativeObj, angleOfs.nativeObj, paddingTL.width, paddingTL.height, paddingBR.width, paddingBR.height);
    }

    /**
     *  Computes gradients and quantized gradient orientations.
     *     @param img Matrix contains the image to be computed
     *     @param grad Matrix of type CV_32FC2 contains computed gradients
     *     @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
     *     @param paddingTL Padding from top-left
     */
    public void computeGradient(Mat img, Mat grad, Mat angleOfs, Size paddingTL) {
        computeGradient_1(nativeObj, img.nativeObj, grad.nativeObj, angleOfs.nativeObj, paddingTL.width, paddingTL.height);
    }

    /**
     *  Computes gradients and quantized gradient orientations.
     *     @param img Matrix contains the image to be computed
     *     @param grad Matrix of type CV_32FC2 contains computed gradients
     *     @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
     */
    public void computeGradient(Mat img, Mat grad, Mat angleOfs) {
        computeGradient_2(nativeObj, img.nativeObj, grad.nativeObj, angleOfs.nativeObj);
    }


    //
    // C++:  void cv::HOGDescriptor::detect(Mat img, vector_Point& foundLocations, vector_double& weights, double hitThreshold = 0, Size winStride = Size(), Size padding = Size(), vector_Point searchLocations = std::vector<Point>())
    //

    /**
     * Performs object detection without a multi-scale window.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
     *     @param weights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     *     @param winStride Window stride. It must be a multiple of block stride.
     *     @param padding Padding
     *     @param searchLocations Vector of Point includes set of requested locations to be evaluated.
     */
    public void detect(Mat img, MatOfPoint foundLocations, MatOfDouble weights, double hitThreshold, Size winStride, Size padding, MatOfPoint searchLocations) {
        Mat foundLocations_mat = foundLocations;
        Mat weights_mat = weights;
        Mat searchLocations_mat = searchLocations;
        detect_0(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, weights_mat.nativeObj, hitThreshold, winStride.width, winStride.height, padding.width, padding.height, searchLocations_mat.nativeObj);
    }

    /**
     * Performs object detection without a multi-scale window.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
     *     @param weights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     *     @param winStride Window stride. It must be a multiple of block stride.
     *     @param padding Padding
     */
    public void detect(Mat img, MatOfPoint foundLocations, MatOfDouble weights, double hitThreshold, Size winStride, Size padding) {
        Mat foundLocations_mat = foundLocations;
        Mat weights_mat = weights;
        detect_1(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, weights_mat.nativeObj, hitThreshold, winStride.width, winStride.height, padding.width, padding.height);
    }

    /**
     * Performs object detection without a multi-scale window.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
     *     @param weights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     *     @param winStride Window stride. It must be a multiple of block stride.
     */
    public void detect(Mat img, MatOfPoint foundLocations, MatOfDouble weights, double hitThreshold, Size winStride) {
        Mat foundLocations_mat = foundLocations;
        Mat weights_mat = weights;
        detect_2(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, weights_mat.nativeObj, hitThreshold, winStride.width, winStride.height);
    }

    /**
     * Performs object detection without a multi-scale window.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
     *     @param weights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     */
    public void detect(Mat img, MatOfPoint foundLocations, MatOfDouble weights, double hitThreshold) {
        Mat foundLocations_mat = foundLocations;
        Mat weights_mat = weights;
        detect_3(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, weights_mat.nativeObj, hitThreshold);
    }

    /**
     * Performs object detection without a multi-scale window.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
     *     @param weights Vector that will contain confidence values for each detected object.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     */
    public void detect(Mat img, MatOfPoint foundLocations, MatOfDouble weights) {
        Mat foundLocations_mat = foundLocations;
        Mat weights_mat = weights;
        detect_4(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, weights_mat.nativeObj);
    }


    //
    // C++:  void cv::HOGDescriptor::detectMultiScale(Mat img, vector_Rect& foundLocations, vector_double& foundWeights, double hitThreshold = 0, Size winStride = Size(), Size padding = Size(), double scale = 1.05, double finalThreshold = 2.0, bool useMeanshiftGrouping = false)
    //

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of rectangles where each rectangle contains the detected object.
     *     @param foundWeights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     *     @param winStride Window stride. It must be a multiple of block stride.
     *     @param padding Padding
     *     @param scale Coefficient of the detection window increase.
     *     @param finalThreshold Final threshold
     *     @param useMeanshiftGrouping indicates grouping algorithm
     */
    public void detectMultiScale(Mat img, MatOfRect foundLocations, MatOfDouble foundWeights, double hitThreshold, Size winStride, Size padding, double scale, double finalThreshold, boolean useMeanshiftGrouping) {
        Mat foundLocations_mat = foundLocations;
        Mat foundWeights_mat = foundWeights;
        detectMultiScale_0(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, foundWeights_mat.nativeObj, hitThreshold, winStride.width, winStride.height, padding.width, padding.height, scale, finalThreshold, useMeanshiftGrouping);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of rectangles where each rectangle contains the detected object.
     *     @param foundWeights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     *     @param winStride Window stride. It must be a multiple of block stride.
     *     @param padding Padding
     *     @param scale Coefficient of the detection window increase.
     *     @param finalThreshold Final threshold
     */
    public void detectMultiScale(Mat img, MatOfRect foundLocations, MatOfDouble foundWeights, double hitThreshold, Size winStride, Size padding, double scale, double finalThreshold) {
        Mat foundLocations_mat = foundLocations;
        Mat foundWeights_mat = foundWeights;
        detectMultiScale_1(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, foundWeights_mat.nativeObj, hitThreshold, winStride.width, winStride.height, padding.width, padding.height, scale, finalThreshold);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of rectangles where each rectangle contains the detected object.
     *     @param foundWeights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     *     @param winStride Window stride. It must be a multiple of block stride.
     *     @param padding Padding
     *     @param scale Coefficient of the detection window increase.
     */
    public void detectMultiScale(Mat img, MatOfRect foundLocations, MatOfDouble foundWeights, double hitThreshold, Size winStride, Size padding, double scale) {
        Mat foundLocations_mat = foundLocations;
        Mat foundWeights_mat = foundWeights;
        detectMultiScale_2(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, foundWeights_mat.nativeObj, hitThreshold, winStride.width, winStride.height, padding.width, padding.height, scale);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of rectangles where each rectangle contains the detected object.
     *     @param foundWeights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     *     @param winStride Window stride. It must be a multiple of block stride.
     *     @param padding Padding
     */
    public void detectMultiScale(Mat img, MatOfRect foundLocations, MatOfDouble foundWeights, double hitThreshold, Size winStride, Size padding) {
        Mat foundLocations_mat = foundLocations;
        Mat foundWeights_mat = foundWeights;
        detectMultiScale_3(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, foundWeights_mat.nativeObj, hitThreshold, winStride.width, winStride.height, padding.width, padding.height);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of rectangles where each rectangle contains the detected object.
     *     @param foundWeights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     *     @param winStride Window stride. It must be a multiple of block stride.
     */
    public void detectMultiScale(Mat img, MatOfRect foundLocations, MatOfDouble foundWeights, double hitThreshold, Size winStride) {
        Mat foundLocations_mat = foundLocations;
        Mat foundWeights_mat = foundWeights;
        detectMultiScale_4(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, foundWeights_mat.nativeObj, hitThreshold, winStride.width, winStride.height);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of rectangles where each rectangle contains the detected object.
     *     @param foundWeights Vector that will contain confidence values for each detected object.
     *     @param hitThreshold Threshold for the distance between features and SVM classifying plane.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     */
    public void detectMultiScale(Mat img, MatOfRect foundLocations, MatOfDouble foundWeights, double hitThreshold) {
        Mat foundLocations_mat = foundLocations;
        Mat foundWeights_mat = foundWeights;
        detectMultiScale_5(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, foundWeights_mat.nativeObj, hitThreshold);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *     @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
     *     @param foundLocations Vector of rectangles where each rectangle contains the detected object.
     *     @param foundWeights Vector that will contain confidence values for each detected object.
     *     Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
     *     But if the free coefficient is omitted (which is allowed), you can specify it manually here.
     */
    public void detectMultiScale(Mat img, MatOfRect foundLocations, MatOfDouble foundWeights) {
        Mat foundLocations_mat = foundLocations;
        Mat foundWeights_mat = foundWeights;
        detectMultiScale_6(nativeObj, img.nativeObj, foundLocations_mat.nativeObj, foundWeights_mat.nativeObj);
    }


    //
    // C++:  void cv::HOGDescriptor::save(String filename, String objname = String())
    //

    /**
     * saves HOGDescriptor parameters and coefficients for the linear SVM classifier to a file
     *     @param filename File name
     *     @param objname Object name
     */
    public void save(String filename, String objname) {
        save_0(nativeObj, filename, objname);
    }

    /**
     * saves HOGDescriptor parameters and coefficients for the linear SVM classifier to a file
     *     @param filename File name
     */
    public void save(String filename) {
        save_1(nativeObj, filename);
    }


    //
    // C++:  void cv::HOGDescriptor::setSVMDetector(Mat svmdetector)
    //

    /**
     * Sets coefficients for the linear SVM classifier.
     *     @param svmdetector coefficients for the linear SVM classifier.
     */
    public void setSVMDetector(Mat svmdetector) {
        setSVMDetector_0(nativeObj, svmdetector.nativeObj);
    }


    //
    // C++: Size HOGDescriptor::winSize
    //

    public Size get_winSize() {
        return new Size(get_winSize_0(nativeObj));
    }


    //
    // C++: Size HOGDescriptor::blockSize
    //

    public Size get_blockSize() {
        return new Size(get_blockSize_0(nativeObj));
    }


    //
    // C++: Size HOGDescriptor::blockStride
    //

    public Size get_blockStride() {
        return new Size(get_blockStride_0(nativeObj));
    }


    //
    // C++: Size HOGDescriptor::cellSize
    //

    public Size get_cellSize() {
        return new Size(get_cellSize_0(nativeObj));
    }


    //
    // C++: int HOGDescriptor::nbins
    //

    public int get_nbins() {
        return get_nbins_0(nativeObj);
    }


    //
    // C++: int HOGDescriptor::derivAperture
    //

    public int get_derivAperture() {
        return get_derivAperture_0(nativeObj);
    }


    //
    // C++: double HOGDescriptor::winSigma
    //

    public double get_winSigma() {
        return get_winSigma_0(nativeObj);
    }


    //
    // C++: HOGDescriptor_HistogramNormType HOGDescriptor::histogramNormType
    //

    public int get_histogramNormType() {
        return get_histogramNormType_0(nativeObj);
    }


    //
    // C++: double HOGDescriptor::L2HysThreshold
    //

    public double get_L2HysThreshold() {
        return get_L2HysThreshold_0(nativeObj);
    }


    //
    // C++: bool HOGDescriptor::gammaCorrection
    //

    public boolean get_gammaCorrection() {
        return get_gammaCorrection_0(nativeObj);
    }


    //
    // C++: vector_float HOGDescriptor::svmDetector
    //

    public MatOfFloat get_svmDetector() {
        return MatOfFloat.fromNativeAddr(get_svmDetector_0(nativeObj));
    }


    //
    // C++: int HOGDescriptor::nlevels
    //

    public int get_nlevels() {
        return get_nlevels_0(nativeObj);
    }


    //
    // C++: bool HOGDescriptor::signedGradient
    //

    public boolean get_signedGradient() {
        return get_signedGradient_0(nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::HOGDescriptor::HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, int _derivAperture = 1, double _winSigma = -1, HOGDescriptor_HistogramNormType _histogramNormType = HOGDescriptor::L2Hys, double _L2HysThreshold = 0.2, bool _gammaCorrection = false, int _nlevels = HOGDescriptor::DEFAULT_NLEVELS, bool _signedGradient = false)
    private static native long HOGDescriptor_0(double _winSize_width, double _winSize_height, double _blockSize_width, double _blockSize_height, double _blockStride_width, double _blockStride_height, double _cellSize_width, double _cellSize_height, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold, boolean _gammaCorrection, int _nlevels, boolean _signedGradient);
    private static native long HOGDescriptor_1(double _winSize_width, double _winSize_height, double _blockSize_width, double _blockSize_height, double _blockStride_width, double _blockStride_height, double _cellSize_width, double _cellSize_height, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold, boolean _gammaCorrection, int _nlevels);
    private static native long HOGDescriptor_2(double _winSize_width, double _winSize_height, double _blockSize_width, double _blockSize_height, double _blockStride_width, double _blockStride_height, double _cellSize_width, double _cellSize_height, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold, boolean _gammaCorrection);
    private static native long HOGDescriptor_3(double _winSize_width, double _winSize_height, double _blockSize_width, double _blockSize_height, double _blockStride_width, double _blockStride_height, double _cellSize_width, double _cellSize_height, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold);
    private static native long HOGDescriptor_4(double _winSize_width, double _winSize_height, double _blockSize_width, double _blockSize_height, double _blockStride_width, double _blockStride_height, double _cellSize_width, double _cellSize_height, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType);
    private static native long HOGDescriptor_5(double _winSize_width, double _winSize_height, double _blockSize_width, double _blockSize_height, double _blockStride_width, double _blockStride_height, double _cellSize_width, double _cellSize_height, int _nbins, int _derivAperture, double _winSigma);
    private static native long HOGDescriptor_6(double _winSize_width, double _winSize_height, double _blockSize_width, double _blockSize_height, double _blockStride_width, double _blockStride_height, double _cellSize_width, double _cellSize_height, int _nbins, int _derivAperture);
    private static native long HOGDescriptor_7(double _winSize_width, double _winSize_height, double _blockSize_width, double _blockSize_height, double _blockStride_width, double _blockStride_height, double _cellSize_width, double _cellSize_height, int _nbins);

    // C++:   cv::HOGDescriptor::HOGDescriptor(String filename)
    private static native long HOGDescriptor_8(String filename);

    // C++:   cv::HOGDescriptor::HOGDescriptor()
    private static native long HOGDescriptor_9();

    // C++:  bool cv::HOGDescriptor::checkDetectorSize()
    private static native boolean checkDetectorSize_0(long nativeObj);

    // C++:  bool cv::HOGDescriptor::load(String filename, String objname = String())
    private static native boolean load_0(long nativeObj, String filename, String objname);
    private static native boolean load_1(long nativeObj, String filename);

    // C++:  double cv::HOGDescriptor::getWinSigma()
    private static native double getWinSigma_0(long nativeObj);

    // C++:  size_t cv::HOGDescriptor::getDescriptorSize()
    private static native long getDescriptorSize_0(long nativeObj);

    // C++: static vector_float cv::HOGDescriptor::getDaimlerPeopleDetector()
    private static native long getDaimlerPeopleDetector_0();

    // C++: static vector_float cv::HOGDescriptor::getDefaultPeopleDetector()
    private static native long getDefaultPeopleDetector_0();

    // C++:  void cv::HOGDescriptor::compute(Mat img, vector_float& descriptors, Size winStride = Size(), Size padding = Size(), vector_Point locations = std::vector<Point>())
    private static native void compute_0(long nativeObj, long img_nativeObj, long descriptors_mat_nativeObj, double winStride_width, double winStride_height, double padding_width, double padding_height, long locations_mat_nativeObj);
    private static native void compute_1(long nativeObj, long img_nativeObj, long descriptors_mat_nativeObj, double winStride_width, double winStride_height, double padding_width, double padding_height);
    private static native void compute_2(long nativeObj, long img_nativeObj, long descriptors_mat_nativeObj, double winStride_width, double winStride_height);
    private static native void compute_3(long nativeObj, long img_nativeObj, long descriptors_mat_nativeObj);

    // C++:  void cv::HOGDescriptor::computeGradient(Mat img, Mat& grad, Mat& angleOfs, Size paddingTL = Size(), Size paddingBR = Size())
    private static native void computeGradient_0(long nativeObj, long img_nativeObj, long grad_nativeObj, long angleOfs_nativeObj, double paddingTL_width, double paddingTL_height, double paddingBR_width, double paddingBR_height);
    private static native void computeGradient_1(long nativeObj, long img_nativeObj, long grad_nativeObj, long angleOfs_nativeObj, double paddingTL_width, double paddingTL_height);
    private static native void computeGradient_2(long nativeObj, long img_nativeObj, long grad_nativeObj, long angleOfs_nativeObj);

    // C++:  void cv::HOGDescriptor::detect(Mat img, vector_Point& foundLocations, vector_double& weights, double hitThreshold = 0, Size winStride = Size(), Size padding = Size(), vector_Point searchLocations = std::vector<Point>())
    private static native void detect_0(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long weights_mat_nativeObj, double hitThreshold, double winStride_width, double winStride_height, double padding_width, double padding_height, long searchLocations_mat_nativeObj);
    private static native void detect_1(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long weights_mat_nativeObj, double hitThreshold, double winStride_width, double winStride_height, double padding_width, double padding_height);
    private static native void detect_2(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long weights_mat_nativeObj, double hitThreshold, double winStride_width, double winStride_height);
    private static native void detect_3(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long weights_mat_nativeObj, double hitThreshold);
    private static native void detect_4(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long weights_mat_nativeObj);

    // C++:  void cv::HOGDescriptor::detectMultiScale(Mat img, vector_Rect& foundLocations, vector_double& foundWeights, double hitThreshold = 0, Size winStride = Size(), Size padding = Size(), double scale = 1.05, double finalThreshold = 2.0, bool useMeanshiftGrouping = false)
    private static native void detectMultiScale_0(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long foundWeights_mat_nativeObj, double hitThreshold, double winStride_width, double winStride_height, double padding_width, double padding_height, double scale, double finalThreshold, boolean useMeanshiftGrouping);
    private static native void detectMultiScale_1(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long foundWeights_mat_nativeObj, double hitThreshold, double winStride_width, double winStride_height, double padding_width, double padding_height, double scale, double finalThreshold);
    private static native void detectMultiScale_2(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long foundWeights_mat_nativeObj, double hitThreshold, double winStride_width, double winStride_height, double padding_width, double padding_height, double scale);
    private static native void detectMultiScale_3(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long foundWeights_mat_nativeObj, double hitThreshold, double winStride_width, double winStride_height, double padding_width, double padding_height);
    private static native void detectMultiScale_4(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long foundWeights_mat_nativeObj, double hitThreshold, double winStride_width, double winStride_height);
    private static native void detectMultiScale_5(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long foundWeights_mat_nativeObj, double hitThreshold);
    private static native void detectMultiScale_6(long nativeObj, long img_nativeObj, long foundLocations_mat_nativeObj, long foundWeights_mat_nativeObj);

    // C++:  void cv::HOGDescriptor::save(String filename, String objname = String())
    private static native void save_0(long nativeObj, String filename, String objname);
    private static native void save_1(long nativeObj, String filename);

    // C++:  void cv::HOGDescriptor::setSVMDetector(Mat svmdetector)
    private static native void setSVMDetector_0(long nativeObj, long svmdetector_nativeObj);

    // C++: Size HOGDescriptor::winSize
    private static native double[] get_winSize_0(long nativeObj);

    // C++: Size HOGDescriptor::blockSize
    private static native double[] get_blockSize_0(long nativeObj);

    // C++: Size HOGDescriptor::blockStride
    private static native double[] get_blockStride_0(long nativeObj);

    // C++: Size HOGDescriptor::cellSize
    private static native double[] get_cellSize_0(long nativeObj);

    // C++: int HOGDescriptor::nbins
    private static native int get_nbins_0(long nativeObj);

    // C++: int HOGDescriptor::derivAperture
    private static native int get_derivAperture_0(long nativeObj);

    // C++: double HOGDescriptor::winSigma
    private static native double get_winSigma_0(long nativeObj);

    // C++: HOGDescriptor_HistogramNormType HOGDescriptor::histogramNormType
    private static native int get_histogramNormType_0(long nativeObj);

    // C++: double HOGDescriptor::L2HysThreshold
    private static native double get_L2HysThreshold_0(long nativeObj);

    // C++: bool HOGDescriptor::gammaCorrection
    private static native boolean get_gammaCorrection_0(long nativeObj);

    // C++: vector_float HOGDescriptor::svmDetector
    private static native long get_svmDetector_0(long nativeObj);

    // C++: int HOGDescriptor::nlevels
    private static native int get_nlevels_0(long nativeObj);

    // C++: bool HOGDescriptor::signedGradient
    private static native boolean get_signedGradient_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
