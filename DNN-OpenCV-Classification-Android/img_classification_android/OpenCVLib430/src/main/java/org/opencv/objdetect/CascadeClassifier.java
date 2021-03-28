//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.objdetect;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Size;
import org.opencv.utils.Converters;

// C++: class CascadeClassifier
/**
 * Cascade classifier class for object detection.
 */
public class CascadeClassifier {

    protected final long nativeObj;
    protected CascadeClassifier(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static CascadeClassifier __fromPtr__(long addr) { return new CascadeClassifier(addr); }

    //
    // C++:   cv::CascadeClassifier::CascadeClassifier(String filename)
    //

    /**
     * Loads a classifier from a file.
     *
     *     @param filename Name of the file from which the classifier is loaded.
     */
    public CascadeClassifier(String filename) {
        nativeObj = CascadeClassifier_0(filename);
    }


    //
    // C++:   cv::CascadeClassifier::CascadeClassifier()
    //

    public CascadeClassifier() {
        nativeObj = CascadeClassifier_1();
    }


    //
    // C++:  Size cv::CascadeClassifier::getOriginalWindowSize()
    //

    public Size getOriginalWindowSize() {
        return new Size(getOriginalWindowSize_0(nativeObj));
    }


    //
    // C++: static bool cv::CascadeClassifier::convert(String oldcascade, String newcascade)
    //

    public static boolean convert(String oldcascade, String newcascade) {
        return convert_0(oldcascade, newcascade);
    }


    //
    // C++:  bool cv::CascadeClassifier::empty()
    //

    /**
     * Checks whether the classifier has been loaded.
     * @return automatically generated
     */
    public boolean empty() {
        return empty_0(nativeObj);
    }


    //
    // C++:  bool cv::CascadeClassifier::isOldFormatCascade()
    //

    public boolean isOldFormatCascade() {
        return isOldFormatCascade_0(nativeObj);
    }


    //
    // C++:  bool cv::CascadeClassifier::load(String filename)
    //

    /**
     * Loads a classifier from a file.
     *
     *     @param filename Name of the file from which the classifier is loaded. The file may contain an old
     *     HAAR classifier trained by the haartraining application or a new cascade classifier trained by the
     *     traincascade application.
     * @return automatically generated
     */
    public boolean load(String filename) {
        return load_0(nativeObj, filename);
    }


    //
    // C++:  bool cv::CascadeClassifier::read(FileNode node)
    //

    // Unknown type 'FileNode' (I), skipping the function


    //
    // C++:  int cv::CascadeClassifier::getFeatureType()
    //

    public int getFeatureType() {
        return getFeatureType_0(nativeObj);
    }


    //
    // C++:  void cv::CascadeClassifier::detectMultiScale(Mat image, vector_Rect& objects, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    //

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
     *     to retain it.
     *     @param flags Parameter with the same meaning for an old cascade as in the function
     *     cvHaarDetectObjects. It is not used for a new cascade.
     *     @param minSize Minimum possible object size. Objects smaller than that are ignored.
     *     @param maxSize Maximum possible object size. Objects larger than that are ignored. If {@code maxSize == minSize} model is evaluated on single scale.
     *
     *     The function is parallelized with the TBB library.
     *
     *     <b>Note:</b>
     * <ul>
     *   <li>
     *           (Python) A face detection example using cascade classifiers can be found at
     *             opencv_source_code/samples/python/facedetect.py
     *   </li>
     * </ul>
     */
    public void detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize) {
        Mat objects_mat = objects;
        detectMultiScale_0(nativeObj, image.nativeObj, objects_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height, maxSize.width, maxSize.height);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
     *     to retain it.
     *     @param flags Parameter with the same meaning for an old cascade as in the function
     *     cvHaarDetectObjects. It is not used for a new cascade.
     *     @param minSize Minimum possible object size. Objects smaller than that are ignored.
     *
     *     The function is parallelized with the TBB library.
     *
     *     <b>Note:</b>
     * <ul>
     *   <li>
     *           (Python) A face detection example using cascade classifiers can be found at
     *             opencv_source_code/samples/python/facedetect.py
     *   </li>
     * </ul>
     */
    public void detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flags, Size minSize) {
        Mat objects_mat = objects;
        detectMultiScale_1(nativeObj, image.nativeObj, objects_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
     *     to retain it.
     *     @param flags Parameter with the same meaning for an old cascade as in the function
     *     cvHaarDetectObjects. It is not used for a new cascade.
     *
     *     The function is parallelized with the TBB library.
     *
     *     <b>Note:</b>
     * <ul>
     *   <li>
     *           (Python) A face detection example using cascade classifiers can be found at
     *             opencv_source_code/samples/python/facedetect.py
     *   </li>
     * </ul>
     */
    public void detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flags) {
        Mat objects_mat = objects;
        detectMultiScale_2(nativeObj, image.nativeObj, objects_mat.nativeObj, scaleFactor, minNeighbors, flags);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
     *     to retain it.
     *     cvHaarDetectObjects. It is not used for a new cascade.
     *
     *     The function is parallelized with the TBB library.
     *
     *     <b>Note:</b>
     * <ul>
     *   <li>
     *           (Python) A face detection example using cascade classifiers can be found at
     *             opencv_source_code/samples/python/facedetect.py
     *   </li>
     * </ul>
     */
    public void detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors) {
        Mat objects_mat = objects;
        detectMultiScale_3(nativeObj, image.nativeObj, objects_mat.nativeObj, scaleFactor, minNeighbors);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     to retain it.
     *     cvHaarDetectObjects. It is not used for a new cascade.
     *
     *     The function is parallelized with the TBB library.
     *
     *     <b>Note:</b>
     * <ul>
     *   <li>
     *           (Python) A face detection example using cascade classifiers can be found at
     *             opencv_source_code/samples/python/facedetect.py
     *   </li>
     * </ul>
     */
    public void detectMultiScale(Mat image, MatOfRect objects, double scaleFactor) {
        Mat objects_mat = objects;
        detectMultiScale_4(nativeObj, image.nativeObj, objects_mat.nativeObj, scaleFactor);
    }

    /**
     * Detects objects of different sizes in the input image. The detected objects are returned as a list
     *     of rectangles.
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     to retain it.
     *     cvHaarDetectObjects. It is not used for a new cascade.
     *
     *     The function is parallelized with the TBB library.
     *
     *     <b>Note:</b>
     * <ul>
     *   <li>
     *           (Python) A face detection example using cascade classifiers can be found at
     *             opencv_source_code/samples/python/facedetect.py
     *   </li>
     * </ul>
     */
    public void detectMultiScale(Mat image, MatOfRect objects) {
        Mat objects_mat = objects;
        detectMultiScale_5(nativeObj, image.nativeObj, objects_mat.nativeObj);
    }


    //
    // C++:  void cv::CascadeClassifier::detectMultiScale(Mat image, vector_Rect& objects, vector_int& numDetections, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    //

    /**
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param numDetections Vector of detection numbers for the corresponding objects. An object's number
     *     of detections is the number of neighboring positively classified rectangles that were joined
     *     together to form the object.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
     *     to retain it.
     *     @param flags Parameter with the same meaning for an old cascade as in the function
     *     cvHaarDetectObjects. It is not used for a new cascade.
     *     @param minSize Minimum possible object size. Objects smaller than that are ignored.
     *     @param maxSize Maximum possible object size. Objects larger than that are ignored. If {@code maxSize == minSize} model is evaluated on single scale.
     */
    public void detectMultiScale2(Mat image, MatOfRect objects, MatOfInt numDetections, double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize) {
        Mat objects_mat = objects;
        Mat numDetections_mat = numDetections;
        detectMultiScale2_0(nativeObj, image.nativeObj, objects_mat.nativeObj, numDetections_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height, maxSize.width, maxSize.height);
    }

    /**
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param numDetections Vector of detection numbers for the corresponding objects. An object's number
     *     of detections is the number of neighboring positively classified rectangles that were joined
     *     together to form the object.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
     *     to retain it.
     *     @param flags Parameter with the same meaning for an old cascade as in the function
     *     cvHaarDetectObjects. It is not used for a new cascade.
     *     @param minSize Minimum possible object size. Objects smaller than that are ignored.
     */
    public void detectMultiScale2(Mat image, MatOfRect objects, MatOfInt numDetections, double scaleFactor, int minNeighbors, int flags, Size minSize) {
        Mat objects_mat = objects;
        Mat numDetections_mat = numDetections;
        detectMultiScale2_1(nativeObj, image.nativeObj, objects_mat.nativeObj, numDetections_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height);
    }

    /**
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param numDetections Vector of detection numbers for the corresponding objects. An object's number
     *     of detections is the number of neighboring positively classified rectangles that were joined
     *     together to form the object.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
     *     to retain it.
     *     @param flags Parameter with the same meaning for an old cascade as in the function
     *     cvHaarDetectObjects. It is not used for a new cascade.
     */
    public void detectMultiScale2(Mat image, MatOfRect objects, MatOfInt numDetections, double scaleFactor, int minNeighbors, int flags) {
        Mat objects_mat = objects;
        Mat numDetections_mat = numDetections;
        detectMultiScale2_2(nativeObj, image.nativeObj, objects_mat.nativeObj, numDetections_mat.nativeObj, scaleFactor, minNeighbors, flags);
    }

    /**
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param numDetections Vector of detection numbers for the corresponding objects. An object's number
     *     of detections is the number of neighboring positively classified rectangles that were joined
     *     together to form the object.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
     *     to retain it.
     *     cvHaarDetectObjects. It is not used for a new cascade.
     */
    public void detectMultiScale2(Mat image, MatOfRect objects, MatOfInt numDetections, double scaleFactor, int minNeighbors) {
        Mat objects_mat = objects;
        Mat numDetections_mat = numDetections;
        detectMultiScale2_3(nativeObj, image.nativeObj, objects_mat.nativeObj, numDetections_mat.nativeObj, scaleFactor, minNeighbors);
    }

    /**
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param numDetections Vector of detection numbers for the corresponding objects. An object's number
     *     of detections is the number of neighboring positively classified rectangles that were joined
     *     together to form the object.
     *     @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
     *     to retain it.
     *     cvHaarDetectObjects. It is not used for a new cascade.
     */
    public void detectMultiScale2(Mat image, MatOfRect objects, MatOfInt numDetections, double scaleFactor) {
        Mat objects_mat = objects;
        Mat numDetections_mat = numDetections;
        detectMultiScale2_4(nativeObj, image.nativeObj, objects_mat.nativeObj, numDetections_mat.nativeObj, scaleFactor);
    }

    /**
     *
     *     @param image Matrix of the type CV_8U containing an image where objects are detected.
     *     @param objects Vector of rectangles where each rectangle contains the detected object, the
     *     rectangles may be partially outside the original image.
     *     @param numDetections Vector of detection numbers for the corresponding objects. An object's number
     *     of detections is the number of neighboring positively classified rectangles that were joined
     *     together to form the object.
     *     to retain it.
     *     cvHaarDetectObjects. It is not used for a new cascade.
     */
    public void detectMultiScale2(Mat image, MatOfRect objects, MatOfInt numDetections) {
        Mat objects_mat = objects;
        Mat numDetections_mat = numDetections;
        detectMultiScale2_5(nativeObj, image.nativeObj, objects_mat.nativeObj, numDetections_mat.nativeObj);
    }


    //
    // C++:  void cv::CascadeClassifier::detectMultiScale(Mat image, vector_Rect& objects, vector_int& rejectLevels, vector_double& levelWeights, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size(), bool outputRejectLevels = false)
    //

    /**
     *
     *     This function allows you to retrieve the final stage decision certainty of classification.
     *     For this, one needs to set {@code outputRejectLevels} on true and provide the {@code rejectLevels} and {@code levelWeights} parameter.
     *     For each resulting detection, {@code levelWeights} will then contain the certainty of classification at the final stage.
     *     This value can then be used to separate strong from weaker classifications.
     *
     *     A code sample on how to use it efficiently can be found below:
     *     <code>
     *     Mat img;
     *     vector&lt;double&gt; weights;
     *     vector&lt;int&gt; levels;
     *     vector&lt;Rect&gt; detections;
     *     CascadeClassifier model("/path/to/your/model.xml");
     *     model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
     *     cerr &lt;&lt; "Detection " &lt;&lt; detections[0] &lt;&lt; " with weight " &lt;&lt; weights[0] &lt;&lt; endl;
     *     </code>
     * @param image automatically generated
     * @param objects automatically generated
     * @param rejectLevels automatically generated
     * @param levelWeights automatically generated
     * @param scaleFactor automatically generated
     * @param minNeighbors automatically generated
     * @param flags automatically generated
     * @param minSize automatically generated
     * @param maxSize automatically generated
     * @param outputRejectLevels automatically generated
     */
    public void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights, double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize, boolean outputRejectLevels) {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_0(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height, maxSize.width, maxSize.height, outputRejectLevels);
    }

    /**
     *
     *     This function allows you to retrieve the final stage decision certainty of classification.
     *     For this, one needs to set {@code outputRejectLevels} on true and provide the {@code rejectLevels} and {@code levelWeights} parameter.
     *     For each resulting detection, {@code levelWeights} will then contain the certainty of classification at the final stage.
     *     This value can then be used to separate strong from weaker classifications.
     *
     *     A code sample on how to use it efficiently can be found below:
     *     <code>
     *     Mat img;
     *     vector&lt;double&gt; weights;
     *     vector&lt;int&gt; levels;
     *     vector&lt;Rect&gt; detections;
     *     CascadeClassifier model("/path/to/your/model.xml");
     *     model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
     *     cerr &lt;&lt; "Detection " &lt;&lt; detections[0] &lt;&lt; " with weight " &lt;&lt; weights[0] &lt;&lt; endl;
     *     </code>
     * @param image automatically generated
     * @param objects automatically generated
     * @param rejectLevels automatically generated
     * @param levelWeights automatically generated
     * @param scaleFactor automatically generated
     * @param minNeighbors automatically generated
     * @param flags automatically generated
     * @param minSize automatically generated
     * @param maxSize automatically generated
     */
    public void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights, double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize) {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_1(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height, maxSize.width, maxSize.height);
    }

    /**
     *
     *     This function allows you to retrieve the final stage decision certainty of classification.
     *     For this, one needs to set {@code outputRejectLevels} on true and provide the {@code rejectLevels} and {@code levelWeights} parameter.
     *     For each resulting detection, {@code levelWeights} will then contain the certainty of classification at the final stage.
     *     This value can then be used to separate strong from weaker classifications.
     *
     *     A code sample on how to use it efficiently can be found below:
     *     <code>
     *     Mat img;
     *     vector&lt;double&gt; weights;
     *     vector&lt;int&gt; levels;
     *     vector&lt;Rect&gt; detections;
     *     CascadeClassifier model("/path/to/your/model.xml");
     *     model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
     *     cerr &lt;&lt; "Detection " &lt;&lt; detections[0] &lt;&lt; " with weight " &lt;&lt; weights[0] &lt;&lt; endl;
     *     </code>
     * @param image automatically generated
     * @param objects automatically generated
     * @param rejectLevels automatically generated
     * @param levelWeights automatically generated
     * @param scaleFactor automatically generated
     * @param minNeighbors automatically generated
     * @param flags automatically generated
     * @param minSize automatically generated
     */
    public void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights, double scaleFactor, int minNeighbors, int flags, Size minSize) {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_2(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height);
    }

    /**
     *
     *     This function allows you to retrieve the final stage decision certainty of classification.
     *     For this, one needs to set {@code outputRejectLevels} on true and provide the {@code rejectLevels} and {@code levelWeights} parameter.
     *     For each resulting detection, {@code levelWeights} will then contain the certainty of classification at the final stage.
     *     This value can then be used to separate strong from weaker classifications.
     *
     *     A code sample on how to use it efficiently can be found below:
     *     <code>
     *     Mat img;
     *     vector&lt;double&gt; weights;
     *     vector&lt;int&gt; levels;
     *     vector&lt;Rect&gt; detections;
     *     CascadeClassifier model("/path/to/your/model.xml");
     *     model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
     *     cerr &lt;&lt; "Detection " &lt;&lt; detections[0] &lt;&lt; " with weight " &lt;&lt; weights[0] &lt;&lt; endl;
     *     </code>
     * @param image automatically generated
     * @param objects automatically generated
     * @param rejectLevels automatically generated
     * @param levelWeights automatically generated
     * @param scaleFactor automatically generated
     * @param minNeighbors automatically generated
     * @param flags automatically generated
     */
    public void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights, double scaleFactor, int minNeighbors, int flags) {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_3(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj, scaleFactor, minNeighbors, flags);
    }

    /**
     *
     *     This function allows you to retrieve the final stage decision certainty of classification.
     *     For this, one needs to set {@code outputRejectLevels} on true and provide the {@code rejectLevels} and {@code levelWeights} parameter.
     *     For each resulting detection, {@code levelWeights} will then contain the certainty of classification at the final stage.
     *     This value can then be used to separate strong from weaker classifications.
     *
     *     A code sample on how to use it efficiently can be found below:
     *     <code>
     *     Mat img;
     *     vector&lt;double&gt; weights;
     *     vector&lt;int&gt; levels;
     *     vector&lt;Rect&gt; detections;
     *     CascadeClassifier model("/path/to/your/model.xml");
     *     model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
     *     cerr &lt;&lt; "Detection " &lt;&lt; detections[0] &lt;&lt; " with weight " &lt;&lt; weights[0] &lt;&lt; endl;
     *     </code>
     * @param image automatically generated
     * @param objects automatically generated
     * @param rejectLevels automatically generated
     * @param levelWeights automatically generated
     * @param scaleFactor automatically generated
     * @param minNeighbors automatically generated
     */
    public void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights, double scaleFactor, int minNeighbors) {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_4(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj, scaleFactor, minNeighbors);
    }

    /**
     *
     *     This function allows you to retrieve the final stage decision certainty of classification.
     *     For this, one needs to set {@code outputRejectLevels} on true and provide the {@code rejectLevels} and {@code levelWeights} parameter.
     *     For each resulting detection, {@code levelWeights} will then contain the certainty of classification at the final stage.
     *     This value can then be used to separate strong from weaker classifications.
     *
     *     A code sample on how to use it efficiently can be found below:
     *     <code>
     *     Mat img;
     *     vector&lt;double&gt; weights;
     *     vector&lt;int&gt; levels;
     *     vector&lt;Rect&gt; detections;
     *     CascadeClassifier model("/path/to/your/model.xml");
     *     model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
     *     cerr &lt;&lt; "Detection " &lt;&lt; detections[0] &lt;&lt; " with weight " &lt;&lt; weights[0] &lt;&lt; endl;
     *     </code>
     * @param image automatically generated
     * @param objects automatically generated
     * @param rejectLevels automatically generated
     * @param levelWeights automatically generated
     * @param scaleFactor automatically generated
     */
    public void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights, double scaleFactor) {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_5(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj, scaleFactor);
    }

    /**
     *
     *     This function allows you to retrieve the final stage decision certainty of classification.
     *     For this, one needs to set {@code outputRejectLevels} on true and provide the {@code rejectLevels} and {@code levelWeights} parameter.
     *     For each resulting detection, {@code levelWeights} will then contain the certainty of classification at the final stage.
     *     This value can then be used to separate strong from weaker classifications.
     *
     *     A code sample on how to use it efficiently can be found below:
     *     <code>
     *     Mat img;
     *     vector&lt;double&gt; weights;
     *     vector&lt;int&gt; levels;
     *     vector&lt;Rect&gt; detections;
     *     CascadeClassifier model("/path/to/your/model.xml");
     *     model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
     *     cerr &lt;&lt; "Detection " &lt;&lt; detections[0] &lt;&lt; " with weight " &lt;&lt; weights[0] &lt;&lt; endl;
     *     </code>
     * @param image automatically generated
     * @param objects automatically generated
     * @param rejectLevels automatically generated
     * @param levelWeights automatically generated
     */
    public void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights) {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_6(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::CascadeClassifier::CascadeClassifier(String filename)
    private static native long CascadeClassifier_0(String filename);

    // C++:   cv::CascadeClassifier::CascadeClassifier()
    private static native long CascadeClassifier_1();

    // C++:  Size cv::CascadeClassifier::getOriginalWindowSize()
    private static native double[] getOriginalWindowSize_0(long nativeObj);

    // C++: static bool cv::CascadeClassifier::convert(String oldcascade, String newcascade)
    private static native boolean convert_0(String oldcascade, String newcascade);

    // C++:  bool cv::CascadeClassifier::empty()
    private static native boolean empty_0(long nativeObj);

    // C++:  bool cv::CascadeClassifier::isOldFormatCascade()
    private static native boolean isOldFormatCascade_0(long nativeObj);

    // C++:  bool cv::CascadeClassifier::load(String filename)
    private static native boolean load_0(long nativeObj, String filename);

    // C++:  int cv::CascadeClassifier::getFeatureType()
    private static native int getFeatureType_0(long nativeObj);

    // C++:  void cv::CascadeClassifier::detectMultiScale(Mat image, vector_Rect& objects, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    private static native void detectMultiScale_0(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height, double maxSize_width, double maxSize_height);
    private static native void detectMultiScale_1(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height);
    private static native void detectMultiScale_2(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, double scaleFactor, int minNeighbors, int flags);
    private static native void detectMultiScale_3(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, double scaleFactor, int minNeighbors);
    private static native void detectMultiScale_4(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, double scaleFactor);
    private static native void detectMultiScale_5(long nativeObj, long image_nativeObj, long objects_mat_nativeObj);

    // C++:  void cv::CascadeClassifier::detectMultiScale(Mat image, vector_Rect& objects, vector_int& numDetections, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    private static native void detectMultiScale2_0(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long numDetections_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height, double maxSize_width, double maxSize_height);
    private static native void detectMultiScale2_1(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long numDetections_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height);
    private static native void detectMultiScale2_2(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long numDetections_mat_nativeObj, double scaleFactor, int minNeighbors, int flags);
    private static native void detectMultiScale2_3(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long numDetections_mat_nativeObj, double scaleFactor, int minNeighbors);
    private static native void detectMultiScale2_4(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long numDetections_mat_nativeObj, double scaleFactor);
    private static native void detectMultiScale2_5(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long numDetections_mat_nativeObj);

    // C++:  void cv::CascadeClassifier::detectMultiScale(Mat image, vector_Rect& objects, vector_int& rejectLevels, vector_double& levelWeights, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size(), bool outputRejectLevels = false)
    private static native void detectMultiScale3_0(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height, double maxSize_width, double maxSize_height, boolean outputRejectLevels);
    private static native void detectMultiScale3_1(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height, double maxSize_width, double maxSize_height);
    private static native void detectMultiScale3_2(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height);
    private static native void detectMultiScale3_3(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj, double scaleFactor, int minNeighbors, int flags);
    private static native void detectMultiScale3_4(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj, double scaleFactor, int minNeighbors);
    private static native void detectMultiScale3_5(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj, double scaleFactor);
    private static native void detectMultiScale3_6(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
