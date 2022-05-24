//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.utils.Converters;

// C++: class Feature2D
/**
 * Abstract base class for 2D image feature detectors and descriptor extractors
 */
public class Feature2D extends Algorithm {

    protected Feature2D(long addr) { super(addr); }

    // internal usage only
    public static Feature2D __fromPtr__(long addr) { return new Feature2D(addr); }

    //
    // C++:  String cv::Feature2D::getDefaultName()
    //

    public String getDefaultName() {
        return getDefaultName_0(nativeObj);
    }


    //
    // C++:  bool cv::Feature2D::empty()
    //

    public boolean empty() {
        return empty_0(nativeObj);
    }


    //
    // C++:  int cv::Feature2D::defaultNorm()
    //

    public int defaultNorm() {
        return defaultNorm_0(nativeObj);
    }


    //
    // C++:  int cv::Feature2D::descriptorSize()
    //

    public int descriptorSize() {
        return descriptorSize_0(nativeObj);
    }


    //
    // C++:  int cv::Feature2D::descriptorType()
    //

    public int descriptorType() {
        return descriptorType_0(nativeObj);
    }


    //
    // C++:  void cv::Feature2D::compute(Mat image, vector_KeyPoint& keypoints, Mat& descriptors)
    //

    /**
     * Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
     *     (second variant).
     *
     *     @param image Image.
     *     @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
     *     computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
     *     with several dominant orientations (for each orientation).
     *     @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
     *     descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
     *     descriptor for keypoint j-th keypoint.
     */
    public void compute(Mat image, MatOfKeyPoint keypoints, Mat descriptors) {
        Mat keypoints_mat = keypoints;
        compute_0(nativeObj, image.nativeObj, keypoints_mat.nativeObj, descriptors.nativeObj);
    }


    //
    // C++:  void cv::Feature2D::compute(vector_Mat images, vector_vector_KeyPoint& keypoints, vector_Mat& descriptors)
    //

    /**
     *
     *
     *     @param images Image set.
     *     @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
     *     computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
     *     with several dominant orientations (for each orientation).
     *     @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
     *     descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
     *     descriptor for keypoint j-th keypoint.
     */
    public void compute(List<Mat> images, List<MatOfKeyPoint> keypoints, List<Mat> descriptors) {
        Mat images_mat = Converters.vector_Mat_to_Mat(images);
        List<Mat> keypoints_tmplm = new ArrayList<Mat>((keypoints != null) ? keypoints.size() : 0);
        Mat keypoints_mat = Converters.vector_vector_KeyPoint_to_Mat(keypoints, keypoints_tmplm);
        Mat descriptors_mat = new Mat();
        compute_1(nativeObj, images_mat.nativeObj, keypoints_mat.nativeObj, descriptors_mat.nativeObj);
        Converters.Mat_to_vector_vector_KeyPoint(keypoints_mat, keypoints);
        keypoints_mat.release();
        Converters.Mat_to_vector_Mat(descriptors_mat, descriptors);
        descriptors_mat.release();
    }


    //
    // C++:  void cv::Feature2D::detect(Mat image, vector_KeyPoint& keypoints, Mat mask = Mat())
    //

    /**
     * Detects keypoints in an image (first variant) or image set (second variant).
     *
     *     @param image Image.
     *     @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
     *     of keypoints detected in images[i] .
     *     @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
     *     matrix with non-zero values in the region of interest.
     */
    public void detect(Mat image, MatOfKeyPoint keypoints, Mat mask) {
        Mat keypoints_mat = keypoints;
        detect_0(nativeObj, image.nativeObj, keypoints_mat.nativeObj, mask.nativeObj);
    }

    /**
     * Detects keypoints in an image (first variant) or image set (second variant).
     *
     *     @param image Image.
     *     @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
     *     of keypoints detected in images[i] .
     *     matrix with non-zero values in the region of interest.
     */
    public void detect(Mat image, MatOfKeyPoint keypoints) {
        Mat keypoints_mat = keypoints;
        detect_1(nativeObj, image.nativeObj, keypoints_mat.nativeObj);
    }


    //
    // C++:  void cv::Feature2D::detect(vector_Mat images, vector_vector_KeyPoint& keypoints, vector_Mat masks = vector_Mat())
    //

    /**
     *
     *     @param images Image set.
     *     @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
     *     of keypoints detected in images[i] .
     *     @param masks Masks for each input image specifying where to look for keypoints (optional).
     *     masks[i] is a mask for images[i].
     */
    public void detect(List<Mat> images, List<MatOfKeyPoint> keypoints, List<Mat> masks) {
        Mat images_mat = Converters.vector_Mat_to_Mat(images);
        Mat keypoints_mat = new Mat();
        Mat masks_mat = Converters.vector_Mat_to_Mat(masks);
        detect_2(nativeObj, images_mat.nativeObj, keypoints_mat.nativeObj, masks_mat.nativeObj);
        Converters.Mat_to_vector_vector_KeyPoint(keypoints_mat, keypoints);
        keypoints_mat.release();
    }

    /**
     *
     *     @param images Image set.
     *     @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
     *     of keypoints detected in images[i] .
     *     masks[i] is a mask for images[i].
     */
    public void detect(List<Mat> images, List<MatOfKeyPoint> keypoints) {
        Mat images_mat = Converters.vector_Mat_to_Mat(images);
        Mat keypoints_mat = new Mat();
        detect_3(nativeObj, images_mat.nativeObj, keypoints_mat.nativeObj);
        Converters.Mat_to_vector_vector_KeyPoint(keypoints_mat, keypoints);
        keypoints_mat.release();
    }


    //
    // C++:  void cv::Feature2D::detectAndCompute(Mat image, Mat mask, vector_KeyPoint& keypoints, Mat& descriptors, bool useProvidedKeypoints = false)
    //

    /**
     * Detects keypoints and computes the descriptors
     * @param image automatically generated
     * @param mask automatically generated
     * @param keypoints automatically generated
     * @param descriptors automatically generated
     * @param useProvidedKeypoints automatically generated
     */
    public void detectAndCompute(Mat image, Mat mask, MatOfKeyPoint keypoints, Mat descriptors, boolean useProvidedKeypoints) {
        Mat keypoints_mat = keypoints;
        detectAndCompute_0(nativeObj, image.nativeObj, mask.nativeObj, keypoints_mat.nativeObj, descriptors.nativeObj, useProvidedKeypoints);
    }

    /**
     * Detects keypoints and computes the descriptors
     * @param image automatically generated
     * @param mask automatically generated
     * @param keypoints automatically generated
     * @param descriptors automatically generated
     */
    public void detectAndCompute(Mat image, Mat mask, MatOfKeyPoint keypoints, Mat descriptors) {
        Mat keypoints_mat = keypoints;
        detectAndCompute_1(nativeObj, image.nativeObj, mask.nativeObj, keypoints_mat.nativeObj, descriptors.nativeObj);
    }


    //
    // C++:  void cv::Feature2D::read(FileNode arg1)
    //

    // Unknown type 'FileNode' (I), skipping the function


    //
    // C++:  void cv::Feature2D::read(String fileName)
    //

    public void read(String fileName) {
        read_0(nativeObj, fileName);
    }


    //
    // C++:  void cv::Feature2D::write(Ptr_FileStorage fs, String name = String())
    //

    // Unknown type 'Ptr_FileStorage' (I), skipping the function


    //
    // C++:  void cv::Feature2D::write(String fileName)
    //

    public void write(String fileName) {
        write_0(nativeObj, fileName);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  String cv::Feature2D::getDefaultName()
    private static native String getDefaultName_0(long nativeObj);

    // C++:  bool cv::Feature2D::empty()
    private static native boolean empty_0(long nativeObj);

    // C++:  int cv::Feature2D::defaultNorm()
    private static native int defaultNorm_0(long nativeObj);

    // C++:  int cv::Feature2D::descriptorSize()
    private static native int descriptorSize_0(long nativeObj);

    // C++:  int cv::Feature2D::descriptorType()
    private static native int descriptorType_0(long nativeObj);

    // C++:  void cv::Feature2D::compute(Mat image, vector_KeyPoint& keypoints, Mat& descriptors)
    private static native void compute_0(long nativeObj, long image_nativeObj, long keypoints_mat_nativeObj, long descriptors_nativeObj);

    // C++:  void cv::Feature2D::compute(vector_Mat images, vector_vector_KeyPoint& keypoints, vector_Mat& descriptors)
    private static native void compute_1(long nativeObj, long images_mat_nativeObj, long keypoints_mat_nativeObj, long descriptors_mat_nativeObj);

    // C++:  void cv::Feature2D::detect(Mat image, vector_KeyPoint& keypoints, Mat mask = Mat())
    private static native void detect_0(long nativeObj, long image_nativeObj, long keypoints_mat_nativeObj, long mask_nativeObj);
    private static native void detect_1(long nativeObj, long image_nativeObj, long keypoints_mat_nativeObj);

    // C++:  void cv::Feature2D::detect(vector_Mat images, vector_vector_KeyPoint& keypoints, vector_Mat masks = vector_Mat())
    private static native void detect_2(long nativeObj, long images_mat_nativeObj, long keypoints_mat_nativeObj, long masks_mat_nativeObj);
    private static native void detect_3(long nativeObj, long images_mat_nativeObj, long keypoints_mat_nativeObj);

    // C++:  void cv::Feature2D::detectAndCompute(Mat image, Mat mask, vector_KeyPoint& keypoints, Mat& descriptors, bool useProvidedKeypoints = false)
    private static native void detectAndCompute_0(long nativeObj, long image_nativeObj, long mask_nativeObj, long keypoints_mat_nativeObj, long descriptors_nativeObj, boolean useProvidedKeypoints);
    private static native void detectAndCompute_1(long nativeObj, long image_nativeObj, long mask_nativeObj, long keypoints_mat_nativeObj, long descriptors_nativeObj);

    // C++:  void cv::Feature2D::read(String fileName)
    private static native void read_0(long nativeObj, String fileName);

    // C++:  void cv::Feature2D::write(String fileName)
    private static native void write_0(long nativeObj, String fileName);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
