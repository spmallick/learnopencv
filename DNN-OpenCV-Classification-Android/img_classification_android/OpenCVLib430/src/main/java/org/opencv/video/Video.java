//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.utils.Converters;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.BackgroundSubtractorMOG2;

// C++: class Video

public class Video {

    private static final int
            CV_LKFLOW_INITIAL_GUESSES = 4,
            CV_LKFLOW_GET_MIN_EIGENVALS = 8;


    // C++: enum <unnamed>
    public static final int
            OPTFLOW_USE_INITIAL_FLOW = 4,
            OPTFLOW_LK_GET_MIN_EIGENVALS = 8,
            OPTFLOW_FARNEBACK_GAUSSIAN = 256,
            MOTION_TRANSLATION = 0,
            MOTION_EUCLIDEAN = 1,
            MOTION_AFFINE = 2,
            MOTION_HOMOGRAPHY = 3;


    //
    // C++:  Mat cv::readOpticalFlow(String path)
    //

    /**
     * Read a .flo file
     *
     *  @param path Path to the file to be loaded
     *
     *  The function readOpticalFlow loads a flow field from a file and returns it as a single matrix.
     *  Resulting Mat has a type CV_32FC2 - floating-point, 2-channel. First channel corresponds to the
     *  flow in the horizontal direction (u), second - vertical (v).
     * @return automatically generated
     */
    public static Mat readOpticalFlow(String path) {
        return new Mat(readOpticalFlow_0(path));
    }


    //
    // C++:  Ptr_BackgroundSubtractorKNN cv::createBackgroundSubtractorKNN(int history = 500, double dist2Threshold = 400.0, bool detectShadows = true)
    //

    /**
     * Creates KNN Background Subtractor
     *
     * @param history Length of the history.
     * @param dist2Threshold Threshold on the squared distance between the pixel and the sample to decide
     * whether a pixel is close to that sample. This parameter does not affect the background update.
     * @param detectShadows If true, the algorithm will detect shadows and mark them. It decreases the
     * speed a bit, so if you do not need this feature, set the parameter to false.
     * @return automatically generated
     */
    public static BackgroundSubtractorKNN createBackgroundSubtractorKNN(int history, double dist2Threshold, boolean detectShadows) {
        return BackgroundSubtractorKNN.__fromPtr__(createBackgroundSubtractorKNN_0(history, dist2Threshold, detectShadows));
    }

    /**
     * Creates KNN Background Subtractor
     *
     * @param history Length of the history.
     * @param dist2Threshold Threshold on the squared distance between the pixel and the sample to decide
     * whether a pixel is close to that sample. This parameter does not affect the background update.
     * speed a bit, so if you do not need this feature, set the parameter to false.
     * @return automatically generated
     */
    public static BackgroundSubtractorKNN createBackgroundSubtractorKNN(int history, double dist2Threshold) {
        return BackgroundSubtractorKNN.__fromPtr__(createBackgroundSubtractorKNN_1(history, dist2Threshold));
    }

    /**
     * Creates KNN Background Subtractor
     *
     * @param history Length of the history.
     * whether a pixel is close to that sample. This parameter does not affect the background update.
     * speed a bit, so if you do not need this feature, set the parameter to false.
     * @return automatically generated
     */
    public static BackgroundSubtractorKNN createBackgroundSubtractorKNN(int history) {
        return BackgroundSubtractorKNN.__fromPtr__(createBackgroundSubtractorKNN_2(history));
    }

    /**
     * Creates KNN Background Subtractor
     *
     * whether a pixel is close to that sample. This parameter does not affect the background update.
     * speed a bit, so if you do not need this feature, set the parameter to false.
     * @return automatically generated
     */
    public static BackgroundSubtractorKNN createBackgroundSubtractorKNN() {
        return BackgroundSubtractorKNN.__fromPtr__(createBackgroundSubtractorKNN_3());
    }


    //
    // C++:  Ptr_BackgroundSubtractorMOG2 cv::createBackgroundSubtractorMOG2(int history = 500, double varThreshold = 16, bool detectShadows = true)
    //

    /**
     * Creates MOG2 Background Subtractor
     *
     * @param history Length of the history.
     * @param varThreshold Threshold on the squared Mahalanobis distance between the pixel and the model
     * to decide whether a pixel is well described by the background model. This parameter does not
     * affect the background update.
     * @param detectShadows If true, the algorithm will detect shadows and mark them. It decreases the
     * speed a bit, so if you do not need this feature, set the parameter to false.
     * @return automatically generated
     */
    public static BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2(int history, double varThreshold, boolean detectShadows) {
        return BackgroundSubtractorMOG2.__fromPtr__(createBackgroundSubtractorMOG2_0(history, varThreshold, detectShadows));
    }

    /**
     * Creates MOG2 Background Subtractor
     *
     * @param history Length of the history.
     * @param varThreshold Threshold on the squared Mahalanobis distance between the pixel and the model
     * to decide whether a pixel is well described by the background model. This parameter does not
     * affect the background update.
     * speed a bit, so if you do not need this feature, set the parameter to false.
     * @return automatically generated
     */
    public static BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2(int history, double varThreshold) {
        return BackgroundSubtractorMOG2.__fromPtr__(createBackgroundSubtractorMOG2_1(history, varThreshold));
    }

    /**
     * Creates MOG2 Background Subtractor
     *
     * @param history Length of the history.
     * to decide whether a pixel is well described by the background model. This parameter does not
     * affect the background update.
     * speed a bit, so if you do not need this feature, set the parameter to false.
     * @return automatically generated
     */
    public static BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2(int history) {
        return BackgroundSubtractorMOG2.__fromPtr__(createBackgroundSubtractorMOG2_2(history));
    }

    /**
     * Creates MOG2 Background Subtractor
     *
     * to decide whether a pixel is well described by the background model. This parameter does not
     * affect the background update.
     * speed a bit, so if you do not need this feature, set the parameter to false.
     * @return automatically generated
     */
    public static BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2() {
        return BackgroundSubtractorMOG2.__fromPtr__(createBackgroundSubtractorMOG2_3());
    }


    //
    // C++:  RotatedRect cv::CamShift(Mat probImage, Rect& window, TermCriteria criteria)
    //

    /**
     * Finds an object center, size, and orientation.
     *
     * @param probImage Back projection of the object histogram. See calcBackProject.
     * @param window Initial search window.
     * @param criteria Stop criteria for the underlying meanShift.
     * returns
     * (in old interfaces) Number of iterations CAMSHIFT took to converge
     * The function implements the CAMSHIFT object tracking algorithm CITE: Bradski98 . First, it finds an
     * object center using meanShift and then adjusts the window size and finds the optimal rotation. The
     * function returns the rotated rectangle structure that includes the object position, size, and
     * orientation. The next position of the search window can be obtained with RotatedRect::boundingRect()
     *
     * See the OpenCV sample camshiftdemo.c that tracks colored objects.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *    (Python) A sample explaining the camshift tracking algorithm can be found at
     *     opencv_source_code/samples/python/camshift.py
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static RotatedRect CamShift(Mat probImage, Rect window, TermCriteria criteria) {
        double[] window_out = new double[4];
        RotatedRect retVal = new RotatedRect(CamShift_0(probImage.nativeObj, window.x, window.y, window.width, window.height, window_out, criteria.type, criteria.maxCount, criteria.epsilon));
        if(window!=null){ window.x = (int)window_out[0]; window.y = (int)window_out[1]; window.width = (int)window_out[2]; window.height = (int)window_out[3]; } 
        return retVal;
    }


    //
    // C++:  bool cv::writeOpticalFlow(String path, Mat flow)
    //

    /**
     * Write a .flo to disk
     *
     *  @param path Path to the file to be written
     *  @param flow Flow field to be stored
     *
     *  The function stores a flow field in a file, returns true on success, false otherwise.
     *  The flow field must be a 2-channel, floating-point matrix (CV_32FC2). First channel corresponds
     *  to the flow in the horizontal direction (u), second - vertical (v).
     * @return automatically generated
     */
    public static boolean writeOpticalFlow(String path, Mat flow) {
        return writeOpticalFlow_0(path, flow.nativeObj);
    }


    //
    // C++:  double cv::computeECC(Mat templateImage, Mat inputImage, Mat inputMask = Mat())
    //

    /**
     * Computes the Enhanced Correlation Coefficient value between two images CITE: EP08 .
     *
     * @param templateImage single-channel template image; CV_8U or CV_32F array.
     * @param inputImage single-channel input image to be warped to provide an image similar to
     *  templateImage, same type as templateImage.
     * @param inputMask An optional mask to indicate valid values of inputImage.
     *
     * SEE:
     * findTransformECC
     * @return automatically generated
     */
    public static double computeECC(Mat templateImage, Mat inputImage, Mat inputMask) {
        return computeECC_0(templateImage.nativeObj, inputImage.nativeObj, inputMask.nativeObj);
    }

    /**
     * Computes the Enhanced Correlation Coefficient value between two images CITE: EP08 .
     *
     * @param templateImage single-channel template image; CV_8U or CV_32F array.
     * @param inputImage single-channel input image to be warped to provide an image similar to
     *  templateImage, same type as templateImage.
     *
     * SEE:
     * findTransformECC
     * @return automatically generated
     */
    public static double computeECC(Mat templateImage, Mat inputImage) {
        return computeECC_1(templateImage.nativeObj, inputImage.nativeObj);
    }


    //
    // C++:  double cv::findTransformECC(Mat templateImage, Mat inputImage, Mat& warpMatrix, int motionType, TermCriteria criteria, Mat inputMask, int gaussFiltSize)
    //

    /**
     * Finds the geometric transform (warp) between two images in terms of the ECC criterion CITE: EP08 .
     *
     * @param templateImage single-channel template image; CV_8U or CV_32F array.
     * @param inputImage single-channel input image which should be warped with the final warpMatrix in
     * order to provide an image similar to templateImage, same type as templateImage.
     * @param warpMatrix floating-point \(2\times 3\) or \(3\times 3\) mapping matrix (warp).
     * @param motionType parameter, specifying the type of motion:
     * <ul>
     *   <li>
     *     <b>MOTION_TRANSLATION</b> sets a translational motion model; warpMatrix is \(2\times 3\) with
     *      the first \(2\times 2\) part being the unity matrix and the rest two parameters being
     *      estimated.
     *   </li>
     *   <li>
     *     <b>MOTION_EUCLIDEAN</b> sets a Euclidean (rigid) transformation as motion model; three
     *      parameters are estimated; warpMatrix is \(2\times 3\).
     *   </li>
     *   <li>
     *     <b>MOTION_AFFINE</b> sets an affine motion model (DEFAULT); six parameters are estimated;
     *      warpMatrix is \(2\times 3\).
     *   </li>
     *   <li>
     *     <b>MOTION_HOMOGRAPHY</b> sets a homography as a motion model; eight parameters are
     *      estimated;\{@code warpMatrix\} is \(3\times 3\).
     * @param criteria parameter, specifying the termination criteria of the ECC algorithm;
     * criteria.epsilon defines the threshold of the increment in the correlation coefficient between two
     * iterations (a negative criteria.epsilon makes criteria.maxcount the only termination criterion).
     * Default values are shown in the declaration above.
     * @param inputMask An optional mask to indicate valid values of inputImage.
     * @param gaussFiltSize An optional value indicating size of gaussian blur filter; (DEFAULT: 5)
     *   </li>
     * </ul>
     *
     * The function estimates the optimum transformation (warpMatrix) with respect to ECC criterion
     * (CITE: EP08), that is
     *
     * \(\texttt{warpMatrix} = \texttt{warpMatrix} = \arg\max_{W} \texttt{ECC}(\texttt{templateImage}(x,y),\texttt{inputImage}(x',y'))\)
     *
     * where
     *
     * \(\begin{bmatrix} x' \\ y' \end{bmatrix} = W \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}\)
     *
     * (the equation holds with homogeneous coordinates for homography). It returns the final enhanced
     * correlation coefficient, that is the correlation coefficient between the template image and the
     * final warped input image. When a \(3\times 3\) matrix is given with motionType =0, 1 or 2, the third
     * row is ignored.
     *
     * Unlike findHomography and estimateRigidTransform, the function findTransformECC implements an
     * area-based alignment that builds on intensity similarities. In essence, the function updates the
     * initial transformation that roughly aligns the images. If this information is missing, the identity
     * warp (unity matrix) is used as an initialization. Note that if images undergo strong
     * displacements/rotations, an initial transformation that roughly aligns the images is necessary
     * (e.g., a simple euclidean/similarity transform that allows for the images showing the same image
     * content approximately). Use inverse warping in the second image to take an image close to the first
     * one, i.e. use the flag WARP_INVERSE_MAP with warpAffine or warpPerspective. See also the OpenCV
     * sample image_alignment.cpp that demonstrates the use of the function. Note that the function throws
     * an exception if algorithm does not converges.
     *
     * SEE:
     * computeECC, estimateAffine2D, estimateAffinePartial2D, findHomography
     * @return automatically generated
     */
    public static double findTransformECC(Mat templateImage, Mat inputImage, Mat warpMatrix, int motionType, TermCriteria criteria, Mat inputMask, int gaussFiltSize) {
        return findTransformECC_0(templateImage.nativeObj, inputImage.nativeObj, warpMatrix.nativeObj, motionType, criteria.type, criteria.maxCount, criteria.epsilon, inputMask.nativeObj, gaussFiltSize);
    }


    //
    // C++:  int cv::buildOpticalFlowPyramid(Mat img, vector_Mat& pyramid, Size winSize, int maxLevel, bool withDerivatives = true, int pyrBorder = BORDER_REFLECT_101, int derivBorder = BORDER_CONSTANT, bool tryReuseInputImage = true)
    //

    /**
     * Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.
     *
     * @param img 8-bit input image.
     * @param pyramid output pyramid.
     * @param winSize window size of optical flow algorithm. Must be not less than winSize argument of
     * calcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels.
     * @param maxLevel 0-based maximal pyramid level number.
     * @param withDerivatives set to precompute gradients for the every pyramid level. If pyramid is
     * constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.
     * @param pyrBorder the border mode for pyramid layers.
     * @param derivBorder the border mode for gradients.
     * @param tryReuseInputImage put ROI of input image into the pyramid if possible. You can pass false
     * to force data copying.
     * @return number of levels in constructed pyramid. Can be less than maxLevel.
     */
    public static int buildOpticalFlowPyramid(Mat img, List<Mat> pyramid, Size winSize, int maxLevel, boolean withDerivatives, int pyrBorder, int derivBorder, boolean tryReuseInputImage) {
        Mat pyramid_mat = new Mat();
        int retVal = buildOpticalFlowPyramid_0(img.nativeObj, pyramid_mat.nativeObj, winSize.width, winSize.height, maxLevel, withDerivatives, pyrBorder, derivBorder, tryReuseInputImage);
        Converters.Mat_to_vector_Mat(pyramid_mat, pyramid);
        pyramid_mat.release();
        return retVal;
    }

    /**
     * Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.
     *
     * @param img 8-bit input image.
     * @param pyramid output pyramid.
     * @param winSize window size of optical flow algorithm. Must be not less than winSize argument of
     * calcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels.
     * @param maxLevel 0-based maximal pyramid level number.
     * @param withDerivatives set to precompute gradients for the every pyramid level. If pyramid is
     * constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.
     * @param pyrBorder the border mode for pyramid layers.
     * @param derivBorder the border mode for gradients.
     * to force data copying.
     * @return number of levels in constructed pyramid. Can be less than maxLevel.
     */
    public static int buildOpticalFlowPyramid(Mat img, List<Mat> pyramid, Size winSize, int maxLevel, boolean withDerivatives, int pyrBorder, int derivBorder) {
        Mat pyramid_mat = new Mat();
        int retVal = buildOpticalFlowPyramid_1(img.nativeObj, pyramid_mat.nativeObj, winSize.width, winSize.height, maxLevel, withDerivatives, pyrBorder, derivBorder);
        Converters.Mat_to_vector_Mat(pyramid_mat, pyramid);
        pyramid_mat.release();
        return retVal;
    }

    /**
     * Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.
     *
     * @param img 8-bit input image.
     * @param pyramid output pyramid.
     * @param winSize window size of optical flow algorithm. Must be not less than winSize argument of
     * calcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels.
     * @param maxLevel 0-based maximal pyramid level number.
     * @param withDerivatives set to precompute gradients for the every pyramid level. If pyramid is
     * constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.
     * @param pyrBorder the border mode for pyramid layers.
     * to force data copying.
     * @return number of levels in constructed pyramid. Can be less than maxLevel.
     */
    public static int buildOpticalFlowPyramid(Mat img, List<Mat> pyramid, Size winSize, int maxLevel, boolean withDerivatives, int pyrBorder) {
        Mat pyramid_mat = new Mat();
        int retVal = buildOpticalFlowPyramid_2(img.nativeObj, pyramid_mat.nativeObj, winSize.width, winSize.height, maxLevel, withDerivatives, pyrBorder);
        Converters.Mat_to_vector_Mat(pyramid_mat, pyramid);
        pyramid_mat.release();
        return retVal;
    }

    /**
     * Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.
     *
     * @param img 8-bit input image.
     * @param pyramid output pyramid.
     * @param winSize window size of optical flow algorithm. Must be not less than winSize argument of
     * calcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels.
     * @param maxLevel 0-based maximal pyramid level number.
     * @param withDerivatives set to precompute gradients for the every pyramid level. If pyramid is
     * constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.
     * to force data copying.
     * @return number of levels in constructed pyramid. Can be less than maxLevel.
     */
    public static int buildOpticalFlowPyramid(Mat img, List<Mat> pyramid, Size winSize, int maxLevel, boolean withDerivatives) {
        Mat pyramid_mat = new Mat();
        int retVal = buildOpticalFlowPyramid_3(img.nativeObj, pyramid_mat.nativeObj, winSize.width, winSize.height, maxLevel, withDerivatives);
        Converters.Mat_to_vector_Mat(pyramid_mat, pyramid);
        pyramid_mat.release();
        return retVal;
    }

    /**
     * Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.
     *
     * @param img 8-bit input image.
     * @param pyramid output pyramid.
     * @param winSize window size of optical flow algorithm. Must be not less than winSize argument of
     * calcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels.
     * @param maxLevel 0-based maximal pyramid level number.
     * constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.
     * to force data copying.
     * @return number of levels in constructed pyramid. Can be less than maxLevel.
     */
    public static int buildOpticalFlowPyramid(Mat img, List<Mat> pyramid, Size winSize, int maxLevel) {
        Mat pyramid_mat = new Mat();
        int retVal = buildOpticalFlowPyramid_4(img.nativeObj, pyramid_mat.nativeObj, winSize.width, winSize.height, maxLevel);
        Converters.Mat_to_vector_Mat(pyramid_mat, pyramid);
        pyramid_mat.release();
        return retVal;
    }


    //
    // C++:  int cv::meanShift(Mat probImage, Rect& window, TermCriteria criteria)
    //

    /**
     * Finds an object on a back projection image.
     *
     * @param probImage Back projection of the object histogram. See calcBackProject for details.
     * @param window Initial search window.
     * @param criteria Stop criteria for the iterative search algorithm.
     * returns
     * :   Number of iterations CAMSHIFT took to converge.
     * The function implements the iterative object search algorithm. It takes the input back projection of
     * an object and the initial position. The mass center in window of the back projection image is
     * computed and the search window center shifts to the mass center. The procedure is repeated until the
     * specified number of iterations criteria.maxCount is done or until the window center shifts by less
     * than criteria.epsilon. The algorithm is used inside CamShift and, unlike CamShift , the search
     * window size or orientation do not change during the search. You can simply pass the output of
     * calcBackProject to this function. But better results can be obtained if you pre-filter the back
     * projection and remove the noise. For example, you can do this by retrieving connected components
     * with findContours , throwing away contours with small area ( contourArea ), and rendering the
     * remaining contours with drawContours.
     * @return automatically generated
     */
    public static int meanShift(Mat probImage, Rect window, TermCriteria criteria) {
        double[] window_out = new double[4];
        int retVal = meanShift_0(probImage.nativeObj, window.x, window.y, window.width, window.height, window_out, criteria.type, criteria.maxCount, criteria.epsilon);
        if(window!=null){ window.x = (int)window_out[0]; window.y = (int)window_out[1]; window.width = (int)window_out[2]; window.height = (int)window_out[3]; } 
        return retVal;
    }


    //
    // C++:  void cv::calcOpticalFlowFarneback(Mat prev, Mat next, Mat& flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags)
    //

    /**
     * Computes a dense optical flow using the Gunnar Farneback's algorithm.
     *
     * @param prev first 8-bit single-channel input image.
     * @param next second input image of the same size and the same type as prev.
     * @param flow computed flow image that has the same size as prev and type CV_32FC2.
     * @param pyr_scale parameter, specifying the image scale (&lt;1) to build pyramids for each image;
     * pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous
     * one.
     * @param levels number of pyramid layers including the initial image; levels=1 means that no extra
     * layers are created and only the original images are used.
     * @param winsize averaging window size; larger values increase the algorithm robustness to image
     * noise and give more chances for fast motion detection, but yield more blurred motion field.
     * @param iterations number of iterations the algorithm does at each pyramid level.
     * @param poly_n size of the pixel neighborhood used to find polynomial expansion in each pixel;
     * larger values mean that the image will be approximated with smoother surfaces, yielding more
     * robust algorithm and more blurred motion field, typically poly_n =5 or 7.
     * @param poly_sigma standard deviation of the Gaussian that is used to smooth derivatives used as a
     * basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a
     * good value would be poly_sigma=1.5.
     * @param flags operation flags that can be a combination of the following:
     * <ul>
     *   <li>
     *     <b>OPTFLOW_USE_INITIAL_FLOW</b> uses the input flow as an initial flow approximation.
     *   </li>
     *   <li>
     *     <b>OPTFLOW_FARNEBACK_GAUSSIAN</b> uses the Gaussian \(\texttt{winsize}\times\texttt{winsize}\)
     *      filter instead of a box filter of the same size for optical flow estimation; usually, this
     *      option gives z more accurate flow than with a box filter, at the cost of lower speed;
     *      normally, winsize for a Gaussian window should be set to a larger value to achieve the same
     *      level of robustness.
     *   </li>
     * </ul>
     *
     * The function finds an optical flow for each prev pixel using the CITE: Farneback2003 algorithm so that
     *
     * \(\texttt{prev} (y,x)  \sim \texttt{next} ( y + \texttt{flow} (y,x)[1],  x + \texttt{flow} (y,x)[0])\)
     *
     * <b>Note:</b>
     *
     * <ul>
     *   <li>
     *    An example using the optical flow algorithm described by Gunnar Farneback can be found at
     *     opencv_source_code/samples/cpp/fback.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the optical flow algorithm described by Gunnar Farneback can be
     *     found at opencv_source_code/samples/python/opt_flow.py
     *   </li>
     * </ul>
     */
    public static void calcOpticalFlowFarneback(Mat prev, Mat next, Mat flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags) {
        calcOpticalFlowFarneback_0(prev.nativeObj, next.nativeObj, flow.nativeObj, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
    }


    //
    // C++:  void cv::calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg, vector_Point2f prevPts, vector_Point2f& nextPts, vector_uchar& status, vector_float& err, Size winSize = Size(21,21), int maxLevel = 3, TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags = 0, double minEigThreshold = 1e-4)
    //

    /**
     * Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with
     * pyramids.
     *
     * @param prevImg first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
     * @param nextImg second input image or pyramid of the same size and the same type as prevImg.
     * @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
     * single-precision floating-point numbers.
     * @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
     * containing the calculated new positions of input features in the second image; when
     * OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
     * @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
     * the flow for the corresponding features has been found, otherwise, it is set to 0.
     * @param err output vector of errors; each element of the vector is set to an error for the
     * corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
     * found then the error is not defined (use the status parameter to find such cases).
     * @param winSize size of the search window at each pyramid level.
     * @param maxLevel 0-based maximal pyramid level number; if set to 0, pyramids are not used (single
     * level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
     * algorithm will use as many levels as pyramids have but no more than maxLevel.
     * @param criteria parameter, specifying the termination criteria of the iterative search algorithm
     * (after the specified maximum number of iterations criteria.maxCount or when the search window
     * moves by less than criteria.epsilon.
     * @param flags operation flags:
     * <ul>
     *   <li>
     *     <b>OPTFLOW_USE_INITIAL_FLOW</b> uses initial estimations, stored in nextPts; if the flag is
     *      not set, then prevPts is copied to nextPts and is considered the initial estimate.
     *   </li>
     *   <li>
     *     <b>OPTFLOW_LK_GET_MIN_EIGENVALS</b> use minimum eigen values as an error measure (see
     *      minEigThreshold description); if the flag is not set, then L1 distance between patches
     *      around the original and a moved point, divided by number of pixels in a window, is used as a
     *      error measure.
     * @param minEigThreshold the algorithm calculates the minimum eigen value of a 2x2 normal matrix of
     * optical flow equations (this matrix is called a spatial gradient matrix in CITE: Bouguet00), divided
     * by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
     * feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
     * performance boost.
     *   </li>
     * </ul>
     *
     * The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
     * CITE: Bouguet00 . The function is parallelized with the TBB library.
     *
     * <b>Note:</b>
     *
     * <ul>
     *   <li>
     *    An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/cpp/lkdemo.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/python/lk_track.py
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade tracker for homography matching can be found at
     *     opencv_source_code/samples/python/lk_homography.py
     *   </li>
     * </ul>
     */
    public static void calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg, MatOfPoint2f prevPts, MatOfPoint2f nextPts, MatOfByte status, MatOfFloat err, Size winSize, int maxLevel, TermCriteria criteria, int flags, double minEigThreshold) {
        Mat prevPts_mat = prevPts;
        Mat nextPts_mat = nextPts;
        Mat status_mat = status;
        Mat err_mat = err;
        calcOpticalFlowPyrLK_0(prevImg.nativeObj, nextImg.nativeObj, prevPts_mat.nativeObj, nextPts_mat.nativeObj, status_mat.nativeObj, err_mat.nativeObj, winSize.width, winSize.height, maxLevel, criteria.type, criteria.maxCount, criteria.epsilon, flags, minEigThreshold);
    }

    /**
     * Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with
     * pyramids.
     *
     * @param prevImg first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
     * @param nextImg second input image or pyramid of the same size and the same type as prevImg.
     * @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
     * single-precision floating-point numbers.
     * @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
     * containing the calculated new positions of input features in the second image; when
     * OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
     * @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
     * the flow for the corresponding features has been found, otherwise, it is set to 0.
     * @param err output vector of errors; each element of the vector is set to an error for the
     * corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
     * found then the error is not defined (use the status parameter to find such cases).
     * @param winSize size of the search window at each pyramid level.
     * @param maxLevel 0-based maximal pyramid level number; if set to 0, pyramids are not used (single
     * level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
     * algorithm will use as many levels as pyramids have but no more than maxLevel.
     * @param criteria parameter, specifying the termination criteria of the iterative search algorithm
     * (after the specified maximum number of iterations criteria.maxCount or when the search window
     * moves by less than criteria.epsilon.
     * @param flags operation flags:
     * <ul>
     *   <li>
     *     <b>OPTFLOW_USE_INITIAL_FLOW</b> uses initial estimations, stored in nextPts; if the flag is
     *      not set, then prevPts is copied to nextPts and is considered the initial estimate.
     *   </li>
     *   <li>
     *     <b>OPTFLOW_LK_GET_MIN_EIGENVALS</b> use minimum eigen values as an error measure (see
     *      minEigThreshold description); if the flag is not set, then L1 distance between patches
     *      around the original and a moved point, divided by number of pixels in a window, is used as a
     *      error measure.
     * optical flow equations (this matrix is called a spatial gradient matrix in CITE: Bouguet00), divided
     * by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
     * feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
     * performance boost.
     *   </li>
     * </ul>
     *
     * The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
     * CITE: Bouguet00 . The function is parallelized with the TBB library.
     *
     * <b>Note:</b>
     *
     * <ul>
     *   <li>
     *    An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/cpp/lkdemo.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/python/lk_track.py
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade tracker for homography matching can be found at
     *     opencv_source_code/samples/python/lk_homography.py
     *   </li>
     * </ul>
     */
    public static void calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg, MatOfPoint2f prevPts, MatOfPoint2f nextPts, MatOfByte status, MatOfFloat err, Size winSize, int maxLevel, TermCriteria criteria, int flags) {
        Mat prevPts_mat = prevPts;
        Mat nextPts_mat = nextPts;
        Mat status_mat = status;
        Mat err_mat = err;
        calcOpticalFlowPyrLK_1(prevImg.nativeObj, nextImg.nativeObj, prevPts_mat.nativeObj, nextPts_mat.nativeObj, status_mat.nativeObj, err_mat.nativeObj, winSize.width, winSize.height, maxLevel, criteria.type, criteria.maxCount, criteria.epsilon, flags);
    }

    /**
     * Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with
     * pyramids.
     *
     * @param prevImg first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
     * @param nextImg second input image or pyramid of the same size and the same type as prevImg.
     * @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
     * single-precision floating-point numbers.
     * @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
     * containing the calculated new positions of input features in the second image; when
     * OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
     * @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
     * the flow for the corresponding features has been found, otherwise, it is set to 0.
     * @param err output vector of errors; each element of the vector is set to an error for the
     * corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
     * found then the error is not defined (use the status parameter to find such cases).
     * @param winSize size of the search window at each pyramid level.
     * @param maxLevel 0-based maximal pyramid level number; if set to 0, pyramids are not used (single
     * level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
     * algorithm will use as many levels as pyramids have but no more than maxLevel.
     * @param criteria parameter, specifying the termination criteria of the iterative search algorithm
     * (after the specified maximum number of iterations criteria.maxCount or when the search window
     * moves by less than criteria.epsilon.
     * <ul>
     *   <li>
     *     <b>OPTFLOW_USE_INITIAL_FLOW</b> uses initial estimations, stored in nextPts; if the flag is
     *      not set, then prevPts is copied to nextPts and is considered the initial estimate.
     *   </li>
     *   <li>
     *     <b>OPTFLOW_LK_GET_MIN_EIGENVALS</b> use minimum eigen values as an error measure (see
     *      minEigThreshold description); if the flag is not set, then L1 distance between patches
     *      around the original and a moved point, divided by number of pixels in a window, is used as a
     *      error measure.
     * optical flow equations (this matrix is called a spatial gradient matrix in CITE: Bouguet00), divided
     * by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
     * feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
     * performance boost.
     *   </li>
     * </ul>
     *
     * The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
     * CITE: Bouguet00 . The function is parallelized with the TBB library.
     *
     * <b>Note:</b>
     *
     * <ul>
     *   <li>
     *    An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/cpp/lkdemo.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/python/lk_track.py
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade tracker for homography matching can be found at
     *     opencv_source_code/samples/python/lk_homography.py
     *   </li>
     * </ul>
     */
    public static void calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg, MatOfPoint2f prevPts, MatOfPoint2f nextPts, MatOfByte status, MatOfFloat err, Size winSize, int maxLevel, TermCriteria criteria) {
        Mat prevPts_mat = prevPts;
        Mat nextPts_mat = nextPts;
        Mat status_mat = status;
        Mat err_mat = err;
        calcOpticalFlowPyrLK_2(prevImg.nativeObj, nextImg.nativeObj, prevPts_mat.nativeObj, nextPts_mat.nativeObj, status_mat.nativeObj, err_mat.nativeObj, winSize.width, winSize.height, maxLevel, criteria.type, criteria.maxCount, criteria.epsilon);
    }

    /**
     * Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with
     * pyramids.
     *
     * @param prevImg first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
     * @param nextImg second input image or pyramid of the same size and the same type as prevImg.
     * @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
     * single-precision floating-point numbers.
     * @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
     * containing the calculated new positions of input features in the second image; when
     * OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
     * @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
     * the flow for the corresponding features has been found, otherwise, it is set to 0.
     * @param err output vector of errors; each element of the vector is set to an error for the
     * corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
     * found then the error is not defined (use the status parameter to find such cases).
     * @param winSize size of the search window at each pyramid level.
     * @param maxLevel 0-based maximal pyramid level number; if set to 0, pyramids are not used (single
     * level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
     * algorithm will use as many levels as pyramids have but no more than maxLevel.
     * (after the specified maximum number of iterations criteria.maxCount or when the search window
     * moves by less than criteria.epsilon.
     * <ul>
     *   <li>
     *     <b>OPTFLOW_USE_INITIAL_FLOW</b> uses initial estimations, stored in nextPts; if the flag is
     *      not set, then prevPts is copied to nextPts and is considered the initial estimate.
     *   </li>
     *   <li>
     *     <b>OPTFLOW_LK_GET_MIN_EIGENVALS</b> use minimum eigen values as an error measure (see
     *      minEigThreshold description); if the flag is not set, then L1 distance between patches
     *      around the original and a moved point, divided by number of pixels in a window, is used as a
     *      error measure.
     * optical flow equations (this matrix is called a spatial gradient matrix in CITE: Bouguet00), divided
     * by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
     * feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
     * performance boost.
     *   </li>
     * </ul>
     *
     * The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
     * CITE: Bouguet00 . The function is parallelized with the TBB library.
     *
     * <b>Note:</b>
     *
     * <ul>
     *   <li>
     *    An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/cpp/lkdemo.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/python/lk_track.py
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade tracker for homography matching can be found at
     *     opencv_source_code/samples/python/lk_homography.py
     *   </li>
     * </ul>
     */
    public static void calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg, MatOfPoint2f prevPts, MatOfPoint2f nextPts, MatOfByte status, MatOfFloat err, Size winSize, int maxLevel) {
        Mat prevPts_mat = prevPts;
        Mat nextPts_mat = nextPts;
        Mat status_mat = status;
        Mat err_mat = err;
        calcOpticalFlowPyrLK_3(prevImg.nativeObj, nextImg.nativeObj, prevPts_mat.nativeObj, nextPts_mat.nativeObj, status_mat.nativeObj, err_mat.nativeObj, winSize.width, winSize.height, maxLevel);
    }

    /**
     * Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with
     * pyramids.
     *
     * @param prevImg first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
     * @param nextImg second input image or pyramid of the same size and the same type as prevImg.
     * @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
     * single-precision floating-point numbers.
     * @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
     * containing the calculated new positions of input features in the second image; when
     * OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
     * @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
     * the flow for the corresponding features has been found, otherwise, it is set to 0.
     * @param err output vector of errors; each element of the vector is set to an error for the
     * corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
     * found then the error is not defined (use the status parameter to find such cases).
     * @param winSize size of the search window at each pyramid level.
     * level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
     * algorithm will use as many levels as pyramids have but no more than maxLevel.
     * (after the specified maximum number of iterations criteria.maxCount or when the search window
     * moves by less than criteria.epsilon.
     * <ul>
     *   <li>
     *     <b>OPTFLOW_USE_INITIAL_FLOW</b> uses initial estimations, stored in nextPts; if the flag is
     *      not set, then prevPts is copied to nextPts and is considered the initial estimate.
     *   </li>
     *   <li>
     *     <b>OPTFLOW_LK_GET_MIN_EIGENVALS</b> use minimum eigen values as an error measure (see
     *      minEigThreshold description); if the flag is not set, then L1 distance between patches
     *      around the original and a moved point, divided by number of pixels in a window, is used as a
     *      error measure.
     * optical flow equations (this matrix is called a spatial gradient matrix in CITE: Bouguet00), divided
     * by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
     * feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
     * performance boost.
     *   </li>
     * </ul>
     *
     * The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
     * CITE: Bouguet00 . The function is parallelized with the TBB library.
     *
     * <b>Note:</b>
     *
     * <ul>
     *   <li>
     *    An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/cpp/lkdemo.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/python/lk_track.py
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade tracker for homography matching can be found at
     *     opencv_source_code/samples/python/lk_homography.py
     *   </li>
     * </ul>
     */
    public static void calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg, MatOfPoint2f prevPts, MatOfPoint2f nextPts, MatOfByte status, MatOfFloat err, Size winSize) {
        Mat prevPts_mat = prevPts;
        Mat nextPts_mat = nextPts;
        Mat status_mat = status;
        Mat err_mat = err;
        calcOpticalFlowPyrLK_4(prevImg.nativeObj, nextImg.nativeObj, prevPts_mat.nativeObj, nextPts_mat.nativeObj, status_mat.nativeObj, err_mat.nativeObj, winSize.width, winSize.height);
    }

    /**
     * Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with
     * pyramids.
     *
     * @param prevImg first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
     * @param nextImg second input image or pyramid of the same size and the same type as prevImg.
     * @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
     * single-precision floating-point numbers.
     * @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
     * containing the calculated new positions of input features in the second image; when
     * OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
     * @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
     * the flow for the corresponding features has been found, otherwise, it is set to 0.
     * @param err output vector of errors; each element of the vector is set to an error for the
     * corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
     * found then the error is not defined (use the status parameter to find such cases).
     * level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
     * algorithm will use as many levels as pyramids have but no more than maxLevel.
     * (after the specified maximum number of iterations criteria.maxCount or when the search window
     * moves by less than criteria.epsilon.
     * <ul>
     *   <li>
     *     <b>OPTFLOW_USE_INITIAL_FLOW</b> uses initial estimations, stored in nextPts; if the flag is
     *      not set, then prevPts is copied to nextPts and is considered the initial estimate.
     *   </li>
     *   <li>
     *     <b>OPTFLOW_LK_GET_MIN_EIGENVALS</b> use minimum eigen values as an error measure (see
     *      minEigThreshold description); if the flag is not set, then L1 distance between patches
     *      around the original and a moved point, divided by number of pixels in a window, is used as a
     *      error measure.
     * optical flow equations (this matrix is called a spatial gradient matrix in CITE: Bouguet00), divided
     * by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
     * feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
     * performance boost.
     *   </li>
     * </ul>
     *
     * The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
     * CITE: Bouguet00 . The function is parallelized with the TBB library.
     *
     * <b>Note:</b>
     *
     * <ul>
     *   <li>
     *    An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/cpp/lkdemo.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade optical flow algorithm can be found at
     *     opencv_source_code/samples/python/lk_track.py
     *   </li>
     *   <li>
     *    (Python) An example using the Lucas-Kanade tracker for homography matching can be found at
     *     opencv_source_code/samples/python/lk_homography.py
     *   </li>
     * </ul>
     */
    public static void calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg, MatOfPoint2f prevPts, MatOfPoint2f nextPts, MatOfByte status, MatOfFloat err) {
        Mat prevPts_mat = prevPts;
        Mat nextPts_mat = nextPts;
        Mat status_mat = status;
        Mat err_mat = err;
        calcOpticalFlowPyrLK_5(prevImg.nativeObj, nextImg.nativeObj, prevPts_mat.nativeObj, nextPts_mat.nativeObj, status_mat.nativeObj, err_mat.nativeObj);
    }




    // C++:  Mat cv::readOpticalFlow(String path)
    private static native long readOpticalFlow_0(String path);

    // C++:  Ptr_BackgroundSubtractorKNN cv::createBackgroundSubtractorKNN(int history = 500, double dist2Threshold = 400.0, bool detectShadows = true)
    private static native long createBackgroundSubtractorKNN_0(int history, double dist2Threshold, boolean detectShadows);
    private static native long createBackgroundSubtractorKNN_1(int history, double dist2Threshold);
    private static native long createBackgroundSubtractorKNN_2(int history);
    private static native long createBackgroundSubtractorKNN_3();

    // C++:  Ptr_BackgroundSubtractorMOG2 cv::createBackgroundSubtractorMOG2(int history = 500, double varThreshold = 16, bool detectShadows = true)
    private static native long createBackgroundSubtractorMOG2_0(int history, double varThreshold, boolean detectShadows);
    private static native long createBackgroundSubtractorMOG2_1(int history, double varThreshold);
    private static native long createBackgroundSubtractorMOG2_2(int history);
    private static native long createBackgroundSubtractorMOG2_3();

    // C++:  RotatedRect cv::CamShift(Mat probImage, Rect& window, TermCriteria criteria)
    private static native double[] CamShift_0(long probImage_nativeObj, int window_x, int window_y, int window_width, int window_height, double[] window_out, int criteria_type, int criteria_maxCount, double criteria_epsilon);

    // C++:  bool cv::writeOpticalFlow(String path, Mat flow)
    private static native boolean writeOpticalFlow_0(String path, long flow_nativeObj);

    // C++:  double cv::computeECC(Mat templateImage, Mat inputImage, Mat inputMask = Mat())
    private static native double computeECC_0(long templateImage_nativeObj, long inputImage_nativeObj, long inputMask_nativeObj);
    private static native double computeECC_1(long templateImage_nativeObj, long inputImage_nativeObj);

    // C++:  double cv::findTransformECC(Mat templateImage, Mat inputImage, Mat& warpMatrix, int motionType, TermCriteria criteria, Mat inputMask, int gaussFiltSize)
    private static native double findTransformECC_0(long templateImage_nativeObj, long inputImage_nativeObj, long warpMatrix_nativeObj, int motionType, int criteria_type, int criteria_maxCount, double criteria_epsilon, long inputMask_nativeObj, int gaussFiltSize);

    // C++:  int cv::buildOpticalFlowPyramid(Mat img, vector_Mat& pyramid, Size winSize, int maxLevel, bool withDerivatives = true, int pyrBorder = BORDER_REFLECT_101, int derivBorder = BORDER_CONSTANT, bool tryReuseInputImage = true)
    private static native int buildOpticalFlowPyramid_0(long img_nativeObj, long pyramid_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel, boolean withDerivatives, int pyrBorder, int derivBorder, boolean tryReuseInputImage);
    private static native int buildOpticalFlowPyramid_1(long img_nativeObj, long pyramid_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel, boolean withDerivatives, int pyrBorder, int derivBorder);
    private static native int buildOpticalFlowPyramid_2(long img_nativeObj, long pyramid_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel, boolean withDerivatives, int pyrBorder);
    private static native int buildOpticalFlowPyramid_3(long img_nativeObj, long pyramid_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel, boolean withDerivatives);
    private static native int buildOpticalFlowPyramid_4(long img_nativeObj, long pyramid_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel);

    // C++:  int cv::meanShift(Mat probImage, Rect& window, TermCriteria criteria)
    private static native int meanShift_0(long probImage_nativeObj, int window_x, int window_y, int window_width, int window_height, double[] window_out, int criteria_type, int criteria_maxCount, double criteria_epsilon);

    // C++:  void cv::calcOpticalFlowFarneback(Mat prev, Mat next, Mat& flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags)
    private static native void calcOpticalFlowFarneback_0(long prev_nativeObj, long next_nativeObj, long flow_nativeObj, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags);

    // C++:  void cv::calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg, vector_Point2f prevPts, vector_Point2f& nextPts, vector_uchar& status, vector_float& err, Size winSize = Size(21,21), int maxLevel = 3, TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags = 0, double minEigThreshold = 1e-4)
    private static native void calcOpticalFlowPyrLK_0(long prevImg_nativeObj, long nextImg_nativeObj, long prevPts_mat_nativeObj, long nextPts_mat_nativeObj, long status_mat_nativeObj, long err_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel, int criteria_type, int criteria_maxCount, double criteria_epsilon, int flags, double minEigThreshold);
    private static native void calcOpticalFlowPyrLK_1(long prevImg_nativeObj, long nextImg_nativeObj, long prevPts_mat_nativeObj, long nextPts_mat_nativeObj, long status_mat_nativeObj, long err_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel, int criteria_type, int criteria_maxCount, double criteria_epsilon, int flags);
    private static native void calcOpticalFlowPyrLK_2(long prevImg_nativeObj, long nextImg_nativeObj, long prevPts_mat_nativeObj, long nextPts_mat_nativeObj, long status_mat_nativeObj, long err_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native void calcOpticalFlowPyrLK_3(long prevImg_nativeObj, long nextImg_nativeObj, long prevPts_mat_nativeObj, long nextPts_mat_nativeObj, long status_mat_nativeObj, long err_mat_nativeObj, double winSize_width, double winSize_height, int maxLevel);
    private static native void calcOpticalFlowPyrLK_4(long prevImg_nativeObj, long nextImg_nativeObj, long prevPts_mat_nativeObj, long nextPts_mat_nativeObj, long status_mat_nativeObj, long err_mat_nativeObj, double winSize_width, double winSize_height);
    private static native void calcOpticalFlowPyrLK_5(long prevImg_nativeObj, long nextImg_nativeObj, long prevPts_mat_nativeObj, long nextPts_mat_nativeObj, long status_mat_nativeObj, long err_mat_nativeObj);

}
