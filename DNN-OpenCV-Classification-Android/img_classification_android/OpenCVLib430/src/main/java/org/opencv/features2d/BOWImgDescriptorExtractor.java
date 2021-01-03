//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.utils.Converters;

// C++: class BOWImgDescriptorExtractor
/**
 * Class to compute an image descriptor using the *bag of visual words*.
 *
 * Such a computation consists of the following steps:
 *
 * 1.  Compute descriptors for a given image and its keypoints set.
 * 2.  Find the nearest visual words from the vocabulary for each keypoint descriptor.
 * 3.  Compute the bag-of-words image descriptor as is a normalized histogram of vocabulary words
 * encountered in the image. The i-th bin of the histogram is a frequency of i-th word of the
 * vocabulary in the given image.
 */
public class BOWImgDescriptorExtractor {

    protected final long nativeObj;
    protected BOWImgDescriptorExtractor(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static BOWImgDescriptorExtractor __fromPtr__(long addr) { return new BOWImgDescriptorExtractor(addr); }

    //
    // C++:   cv::BOWImgDescriptorExtractor::BOWImgDescriptorExtractor(Ptr_DescriptorExtractor dextractor, Ptr_DescriptorMatcher dmatcher)
    //

    // Unknown type 'Ptr_DescriptorExtractor' (I), skipping the function


    //
    // C++:  Mat cv::BOWImgDescriptorExtractor::getVocabulary()
    //

    /**
     * Returns the set vocabulary.
     * @return automatically generated
     */
    public Mat getVocabulary() {
        return new Mat(getVocabulary_0(nativeObj));
    }


    //
    // C++:  int cv::BOWImgDescriptorExtractor::descriptorSize()
    //

    /**
     * Returns an image descriptor size if the vocabulary is set. Otherwise, it returns 0.
     * @return automatically generated
     */
    public int descriptorSize() {
        return descriptorSize_0(nativeObj);
    }


    //
    // C++:  int cv::BOWImgDescriptorExtractor::descriptorType()
    //

    /**
     * Returns an image descriptor type.
     * @return automatically generated
     */
    public int descriptorType() {
        return descriptorType_0(nativeObj);
    }


    //
    // C++:  void cv::BOWImgDescriptorExtractor::compute2(Mat image, vector_KeyPoint keypoints, Mat& imgDescriptor)
    //

    /**
     *
     *     @param imgDescriptor Computed output image descriptor.
     *     pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary)
     *     returned if it is non-zero.
     * @param image automatically generated
     * @param keypoints automatically generated
     */
    public void compute(Mat image, MatOfKeyPoint keypoints, Mat imgDescriptor) {
        Mat keypoints_mat = keypoints;
        compute_0(nativeObj, image.nativeObj, keypoints_mat.nativeObj, imgDescriptor.nativeObj);
    }


    //
    // C++:  void cv::BOWImgDescriptorExtractor::setVocabulary(Mat vocabulary)
    //

    /**
     * Sets a visual vocabulary.
     *
     *     @param vocabulary Vocabulary (can be trained using the inheritor of BOWTrainer ). Each row of the
     *     vocabulary is a visual word (cluster center).
     */
    public void setVocabulary(Mat vocabulary) {
        setVocabulary_0(nativeObj, vocabulary.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::BOWImgDescriptorExtractor::getVocabulary()
    private static native long getVocabulary_0(long nativeObj);

    // C++:  int cv::BOWImgDescriptorExtractor::descriptorSize()
    private static native int descriptorSize_0(long nativeObj);

    // C++:  int cv::BOWImgDescriptorExtractor::descriptorType()
    private static native int descriptorType_0(long nativeObj);

    // C++:  void cv::BOWImgDescriptorExtractor::compute2(Mat image, vector_KeyPoint keypoints, Mat& imgDescriptor)
    private static native void compute_0(long nativeObj, long image_nativeObj, long keypoints_mat_nativeObj, long imgDescriptor_nativeObj);

    // C++:  void cv::BOWImgDescriptorExtractor::setVocabulary(Mat vocabulary)
    private static native void setVocabulary_0(long nativeObj, long vocabulary_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
