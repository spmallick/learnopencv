//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

// C++: class BOWTrainer
/**
 * Abstract base class for training the *bag of visual words* vocabulary from a set of descriptors.
 *
 * For details, see, for example, *Visual Categorization with Bags of Keypoints* by Gabriella Csurka,
 * Christopher R. Dance, Lixin Fan, Jutta Willamowski, Cedric Bray, 2004. :
 */
public class BOWTrainer {

    protected final long nativeObj;
    protected BOWTrainer(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static BOWTrainer __fromPtr__(long addr) { return new BOWTrainer(addr); }

    //
    // C++:  Mat cv::BOWTrainer::cluster(Mat descriptors)
    //

    /**
     * Clusters train descriptors.
     *
     *     @param descriptors Descriptors to cluster. Each row of the descriptors matrix is a descriptor.
     *     Descriptors are not added to the inner train descriptor set.
     *
     *     The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first
     *     variant of the method, train descriptors stored in the object are clustered. In the second variant,
     *     input descriptors are clustered.
     * @return automatically generated
     */
    public Mat cluster(Mat descriptors) {
        return new Mat(cluster_0(nativeObj, descriptors.nativeObj));
    }


    //
    // C++:  Mat cv::BOWTrainer::cluster()
    //

    public Mat cluster() {
        return new Mat(cluster_1(nativeObj));
    }


    //
    // C++:  int cv::BOWTrainer::descriptorsCount()
    //

    /**
     * Returns the count of all descriptors stored in the training set.
     * @return automatically generated
     */
    public int descriptorsCount() {
        return descriptorsCount_0(nativeObj);
    }


    //
    // C++:  vector_Mat cv::BOWTrainer::getDescriptors()
    //

    /**
     * Returns a training set of descriptors.
     * @return automatically generated
     */
    public List<Mat> getDescriptors() {
        List<Mat> retVal = new ArrayList<Mat>();
        Mat retValMat = new Mat(getDescriptors_0(nativeObj));
        Converters.Mat_to_vector_Mat(retValMat, retVal);
        return retVal;
    }


    //
    // C++:  void cv::BOWTrainer::add(Mat descriptors)
    //

    /**
     * Adds descriptors to a training set.
     *
     *     @param descriptors Descriptors to add to a training set. Each row of the descriptors matrix is a
     *     descriptor.
     *
     *     The training set is clustered using clustermethod to construct the vocabulary.
     */
    public void add(Mat descriptors) {
        add_0(nativeObj, descriptors.nativeObj);
    }


    //
    // C++:  void cv::BOWTrainer::clear()
    //

    public void clear() {
        clear_0(nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::BOWTrainer::cluster(Mat descriptors)
    private static native long cluster_0(long nativeObj, long descriptors_nativeObj);

    // C++:  Mat cv::BOWTrainer::cluster()
    private static native long cluster_1(long nativeObj);

    // C++:  int cv::BOWTrainer::descriptorsCount()
    private static native int descriptorsCount_0(long nativeObj);

    // C++:  vector_Mat cv::BOWTrainer::getDescriptors()
    private static native long getDescriptors_0(long nativeObj);

    // C++:  void cv::BOWTrainer::add(Mat descriptors)
    private static native void add_0(long nativeObj, long descriptors_nativeObj);

    // C++:  void cv::BOWTrainer::clear()
    private static native void clear_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
