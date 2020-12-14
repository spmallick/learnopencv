//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.ml.KNearest;
import org.opencv.ml.StatModel;

// C++: class KNearest
/**
 * The class implements K-Nearest Neighbors model
 *
 * SEE: REF: ml_intro_knn
 */
public class KNearest extends StatModel {

    protected KNearest(long addr) { super(addr); }

    // internal usage only
    public static KNearest __fromPtr__(long addr) { return new KNearest(addr); }

    // C++: enum Types
    public static final int
            BRUTE_FORCE = 1,
            KDTREE = 2;


    //
    // C++: static Ptr_KNearest cv::ml::KNearest::create()
    //

    /**
     * Creates the empty model
     *
     *     The static method creates empty %KNearest classifier. It should be then trained using StatModel::train method.
     * @return automatically generated
     */
    public static KNearest create() {
        return KNearest.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_KNearest cv::ml::KNearest::load(String filepath)
    //

    /**
     * Loads and creates a serialized knearest from a file
     *
     * Use KNearest::save to serialize and store an KNearest to disk.
     * Load the KNearest from this file again, by calling this function with the path to the file.
     *
     * @param filepath path to serialized KNearest
     * @return automatically generated
     */
    public static KNearest load(String filepath) {
        return KNearest.__fromPtr__(load_0(filepath));
    }


    //
    // C++:  bool cv::ml::KNearest::getIsClassifier()
    //

    /**
     * SEE: setIsClassifier
     * @return automatically generated
     */
    public boolean getIsClassifier() {
        return getIsClassifier_0(nativeObj);
    }


    //
    // C++:  float cv::ml::KNearest::findNearest(Mat samples, int k, Mat& results, Mat& neighborResponses = Mat(), Mat& dist = Mat())
    //

    /**
     * Finds the neighbors and predicts responses for input vectors.
     *
     *     @param samples Input samples stored by rows. It is a single-precision floating-point matrix of
     *         {@code &lt;number_of_samples&gt; * k} size.
     *     @param k Number of used nearest neighbors. Should be greater than 1.
     *     @param results Vector with results of prediction (regression or classification) for each input
     *         sample. It is a single-precision floating-point vector with {@code &lt;number_of_samples&gt;} elements.
     *     @param neighborResponses Optional output values for corresponding neighbors. It is a single-
     *         precision floating-point matrix of {@code &lt;number_of_samples&gt; * k} size.
     *     @param dist Optional output distances from the input vectors to the corresponding neighbors. It
     *         is a single-precision floating-point matrix of {@code &lt;number_of_samples&gt; * k} size.
     *
     *     For each input vector (a row of the matrix samples), the method finds the k nearest neighbors.
     *     In case of regression, the predicted result is a mean value of the particular vector's neighbor
     *     responses. In case of classification, the class is determined by voting.
     *
     *     For each input vector, the neighbors are sorted by their distances to the vector.
     *
     *     In case of C++ interface you can use output pointers to empty matrices and the function will
     *     allocate memory itself.
     *
     *     If only a single input vector is passed, all output matrices are optional and the predicted
     *     value is returned by the method.
     *
     *     The function is parallelized with the TBB library.
     * @return automatically generated
     */
    public float findNearest(Mat samples, int k, Mat results, Mat neighborResponses, Mat dist) {
        return findNearest_0(nativeObj, samples.nativeObj, k, results.nativeObj, neighborResponses.nativeObj, dist.nativeObj);
    }

    /**
     * Finds the neighbors and predicts responses for input vectors.
     *
     *     @param samples Input samples stored by rows. It is a single-precision floating-point matrix of
     *         {@code &lt;number_of_samples&gt; * k} size.
     *     @param k Number of used nearest neighbors. Should be greater than 1.
     *     @param results Vector with results of prediction (regression or classification) for each input
     *         sample. It is a single-precision floating-point vector with {@code &lt;number_of_samples&gt;} elements.
     *     @param neighborResponses Optional output values for corresponding neighbors. It is a single-
     *         precision floating-point matrix of {@code &lt;number_of_samples&gt; * k} size.
     *         is a single-precision floating-point matrix of {@code &lt;number_of_samples&gt; * k} size.
     *
     *     For each input vector (a row of the matrix samples), the method finds the k nearest neighbors.
     *     In case of regression, the predicted result is a mean value of the particular vector's neighbor
     *     responses. In case of classification, the class is determined by voting.
     *
     *     For each input vector, the neighbors are sorted by their distances to the vector.
     *
     *     In case of C++ interface you can use output pointers to empty matrices and the function will
     *     allocate memory itself.
     *
     *     If only a single input vector is passed, all output matrices are optional and the predicted
     *     value is returned by the method.
     *
     *     The function is parallelized with the TBB library.
     * @return automatically generated
     */
    public float findNearest(Mat samples, int k, Mat results, Mat neighborResponses) {
        return findNearest_1(nativeObj, samples.nativeObj, k, results.nativeObj, neighborResponses.nativeObj);
    }

    /**
     * Finds the neighbors and predicts responses for input vectors.
     *
     *     @param samples Input samples stored by rows. It is a single-precision floating-point matrix of
     *         {@code &lt;number_of_samples&gt; * k} size.
     *     @param k Number of used nearest neighbors. Should be greater than 1.
     *     @param results Vector with results of prediction (regression or classification) for each input
     *         sample. It is a single-precision floating-point vector with {@code &lt;number_of_samples&gt;} elements.
     *         precision floating-point matrix of {@code &lt;number_of_samples&gt; * k} size.
     *         is a single-precision floating-point matrix of {@code &lt;number_of_samples&gt; * k} size.
     *
     *     For each input vector (a row of the matrix samples), the method finds the k nearest neighbors.
     *     In case of regression, the predicted result is a mean value of the particular vector's neighbor
     *     responses. In case of classification, the class is determined by voting.
     *
     *     For each input vector, the neighbors are sorted by their distances to the vector.
     *
     *     In case of C++ interface you can use output pointers to empty matrices and the function will
     *     allocate memory itself.
     *
     *     If only a single input vector is passed, all output matrices are optional and the predicted
     *     value is returned by the method.
     *
     *     The function is parallelized with the TBB library.
     * @return automatically generated
     */
    public float findNearest(Mat samples, int k, Mat results) {
        return findNearest_2(nativeObj, samples.nativeObj, k, results.nativeObj);
    }


    //
    // C++:  int cv::ml::KNearest::getAlgorithmType()
    //

    /**
     * SEE: setAlgorithmType
     * @return automatically generated
     */
    public int getAlgorithmType() {
        return getAlgorithmType_0(nativeObj);
    }


    //
    // C++:  int cv::ml::KNearest::getDefaultK()
    //

    /**
     * SEE: setDefaultK
     * @return automatically generated
     */
    public int getDefaultK() {
        return getDefaultK_0(nativeObj);
    }


    //
    // C++:  int cv::ml::KNearest::getEmax()
    //

    /**
     * SEE: setEmax
     * @return automatically generated
     */
    public int getEmax() {
        return getEmax_0(nativeObj);
    }


    //
    // C++:  void cv::ml::KNearest::setAlgorithmType(int val)
    //

    /**
     *  getAlgorithmType SEE: getAlgorithmType
     * @param val automatically generated
     */
    public void setAlgorithmType(int val) {
        setAlgorithmType_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::KNearest::setDefaultK(int val)
    //

    /**
     *  getDefaultK SEE: getDefaultK
     * @param val automatically generated
     */
    public void setDefaultK(int val) {
        setDefaultK_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::KNearest::setEmax(int val)
    //

    /**
     *  getEmax SEE: getEmax
     * @param val automatically generated
     */
    public void setEmax(int val) {
        setEmax_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::KNearest::setIsClassifier(bool val)
    //

    /**
     *  getIsClassifier SEE: getIsClassifier
     * @param val automatically generated
     */
    public void setIsClassifier(boolean val) {
        setIsClassifier_0(nativeObj, val);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_KNearest cv::ml::KNearest::create()
    private static native long create_0();

    // C++: static Ptr_KNearest cv::ml::KNearest::load(String filepath)
    private static native long load_0(String filepath);

    // C++:  bool cv::ml::KNearest::getIsClassifier()
    private static native boolean getIsClassifier_0(long nativeObj);

    // C++:  float cv::ml::KNearest::findNearest(Mat samples, int k, Mat& results, Mat& neighborResponses = Mat(), Mat& dist = Mat())
    private static native float findNearest_0(long nativeObj, long samples_nativeObj, int k, long results_nativeObj, long neighborResponses_nativeObj, long dist_nativeObj);
    private static native float findNearest_1(long nativeObj, long samples_nativeObj, int k, long results_nativeObj, long neighborResponses_nativeObj);
    private static native float findNearest_2(long nativeObj, long samples_nativeObj, int k, long results_nativeObj);

    // C++:  int cv::ml::KNearest::getAlgorithmType()
    private static native int getAlgorithmType_0(long nativeObj);

    // C++:  int cv::ml::KNearest::getDefaultK()
    private static native int getDefaultK_0(long nativeObj);

    // C++:  int cv::ml::KNearest::getEmax()
    private static native int getEmax_0(long nativeObj);

    // C++:  void cv::ml::KNearest::setAlgorithmType(int val)
    private static native void setAlgorithmType_0(long nativeObj, int val);

    // C++:  void cv::ml::KNearest::setDefaultK(int val)
    private static native void setDefaultK_0(long nativeObj, int val);

    // C++:  void cv::ml::KNearest::setEmax(int val)
    private static native void setEmax_0(long nativeObj, int val);

    // C++:  void cv::ml::KNearest::setIsClassifier(bool val)
    private static native void setIsClassifier_0(long nativeObj, boolean val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
