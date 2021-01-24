//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.BOWTrainer;

// C++: class BOWKMeansTrainer
/**
 * kmeans -based class to train visual vocabulary using the *bag of visual words* approach. :
 */
public class BOWKMeansTrainer extends BOWTrainer {

    protected BOWKMeansTrainer(long addr) { super(addr); }

    // internal usage only
    public static BOWKMeansTrainer __fromPtr__(long addr) { return new BOWKMeansTrainer(addr); }

    //
    // C++:   cv::BOWKMeansTrainer::BOWKMeansTrainer(int clusterCount, TermCriteria termcrit = TermCriteria(), int attempts = 3, int flags = KMEANS_PP_CENTERS)
    //

    /**
     * The constructor.
     *
     *     SEE: cv::kmeans
     * @param clusterCount automatically generated
     * @param termcrit automatically generated
     * @param attempts automatically generated
     * @param flags automatically generated
     */
    public BOWKMeansTrainer(int clusterCount, TermCriteria termcrit, int attempts, int flags) {
        super(BOWKMeansTrainer_0(clusterCount, termcrit.type, termcrit.maxCount, termcrit.epsilon, attempts, flags));
    }

    /**
     * The constructor.
     *
     *     SEE: cv::kmeans
     * @param clusterCount automatically generated
     * @param termcrit automatically generated
     * @param attempts automatically generated
     */
    public BOWKMeansTrainer(int clusterCount, TermCriteria termcrit, int attempts) {
        super(BOWKMeansTrainer_1(clusterCount, termcrit.type, termcrit.maxCount, termcrit.epsilon, attempts));
    }

    /**
     * The constructor.
     *
     *     SEE: cv::kmeans
     * @param clusterCount automatically generated
     * @param termcrit automatically generated
     */
    public BOWKMeansTrainer(int clusterCount, TermCriteria termcrit) {
        super(BOWKMeansTrainer_2(clusterCount, termcrit.type, termcrit.maxCount, termcrit.epsilon));
    }

    /**
     * The constructor.
     *
     *     SEE: cv::kmeans
     * @param clusterCount automatically generated
     */
    public BOWKMeansTrainer(int clusterCount) {
        super(BOWKMeansTrainer_3(clusterCount));
    }


    //
    // C++:  Mat cv::BOWKMeansTrainer::cluster(Mat descriptors)
    //

    public Mat cluster(Mat descriptors) {
        return new Mat(cluster_0(nativeObj, descriptors.nativeObj));
    }


    //
    // C++:  Mat cv::BOWKMeansTrainer::cluster()
    //

    public Mat cluster() {
        return new Mat(cluster_1(nativeObj));
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::BOWKMeansTrainer::BOWKMeansTrainer(int clusterCount, TermCriteria termcrit = TermCriteria(), int attempts = 3, int flags = KMEANS_PP_CENTERS)
    private static native long BOWKMeansTrainer_0(int clusterCount, int termcrit_type, int termcrit_maxCount, double termcrit_epsilon, int attempts, int flags);
    private static native long BOWKMeansTrainer_1(int clusterCount, int termcrit_type, int termcrit_maxCount, double termcrit_epsilon, int attempts);
    private static native long BOWKMeansTrainer_2(int clusterCount, int termcrit_type, int termcrit_maxCount, double termcrit_epsilon);
    private static native long BOWKMeansTrainer_3(int clusterCount);

    // C++:  Mat cv::BOWKMeansTrainer::cluster(Mat descriptors)
    private static native long cluster_0(long nativeObj, long descriptors_nativeObj);

    // C++:  Mat cv::BOWKMeansTrainer::cluster()
    private static native long cluster_1(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
