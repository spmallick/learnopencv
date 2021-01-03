//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.LogisticRegression;
import org.opencv.ml.StatModel;

// C++: class LogisticRegression
/**
 * Implements Logistic Regression classifier.
 *
 * SEE: REF: ml_intro_lr
 */
public class LogisticRegression extends StatModel {

    protected LogisticRegression(long addr) { super(addr); }

    // internal usage only
    public static LogisticRegression __fromPtr__(long addr) { return new LogisticRegression(addr); }

    // C++: enum RegKinds
    public static final int
            REG_DISABLE = -1,
            REG_L1 = 0,
            REG_L2 = 1;


    // C++: enum Methods
    public static final int
            BATCH = 0,
            MINI_BATCH = 1;


    //
    // C++:  Mat cv::ml::LogisticRegression::get_learnt_thetas()
    //

    /**
     * This function returns the trained parameters arranged across rows.
     *
     *     For a two class classification problem, it returns a row matrix. It returns learnt parameters of
     *     the Logistic Regression as a matrix of type CV_32F.
     * @return automatically generated
     */
    public Mat get_learnt_thetas() {
        return new Mat(get_learnt_thetas_0(nativeObj));
    }


    //
    // C++: static Ptr_LogisticRegression cv::ml::LogisticRegression::create()
    //

    /**
     * Creates empty model.
     *
     *     Creates Logistic Regression model with parameters given.
     * @return automatically generated
     */
    public static LogisticRegression create() {
        return LogisticRegression.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_LogisticRegression cv::ml::LogisticRegression::load(String filepath, String nodeName = String())
    //

    /**
     * Loads and creates a serialized LogisticRegression from a file
     *
     * Use LogisticRegression::save to serialize and store an LogisticRegression to disk.
     * Load the LogisticRegression from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized LogisticRegression
     * @param nodeName name of node containing the classifier
     * @return automatically generated
     */
    public static LogisticRegression load(String filepath, String nodeName) {
        return LogisticRegression.__fromPtr__(load_0(filepath, nodeName));
    }

    /**
     * Loads and creates a serialized LogisticRegression from a file
     *
     * Use LogisticRegression::save to serialize and store an LogisticRegression to disk.
     * Load the LogisticRegression from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized LogisticRegression
     * @return automatically generated
     */
    public static LogisticRegression load(String filepath) {
        return LogisticRegression.__fromPtr__(load_1(filepath));
    }


    //
    // C++:  TermCriteria cv::ml::LogisticRegression::getTermCriteria()
    //

    /**
     * SEE: setTermCriteria
     * @return automatically generated
     */
    public TermCriteria getTermCriteria() {
        return new TermCriteria(getTermCriteria_0(nativeObj));
    }


    //
    // C++:  double cv::ml::LogisticRegression::getLearningRate()
    //

    /**
     * SEE: setLearningRate
     * @return automatically generated
     */
    public double getLearningRate() {
        return getLearningRate_0(nativeObj);
    }


    //
    // C++:  float cv::ml::LogisticRegression::predict(Mat samples, Mat& results = Mat(), int flags = 0)
    //

    /**
     * Predicts responses for input samples and returns a float type.
     *
     *     @param samples The input data for the prediction algorithm. Matrix [m x n], where each row
     *         contains variables (features) of one object being classified. Should have data type CV_32F.
     *     @param results Predicted labels as a column matrix of type CV_32S.
     *     @param flags Not used.
     * @return automatically generated
     */
    public float predict(Mat samples, Mat results, int flags) {
        return predict_0(nativeObj, samples.nativeObj, results.nativeObj, flags);
    }

    /**
     * Predicts responses for input samples and returns a float type.
     *
     *     @param samples The input data for the prediction algorithm. Matrix [m x n], where each row
     *         contains variables (features) of one object being classified. Should have data type CV_32F.
     *     @param results Predicted labels as a column matrix of type CV_32S.
     * @return automatically generated
     */
    public float predict(Mat samples, Mat results) {
        return predict_1(nativeObj, samples.nativeObj, results.nativeObj);
    }

    /**
     * Predicts responses for input samples and returns a float type.
     *
     *     @param samples The input data for the prediction algorithm. Matrix [m x n], where each row
     *         contains variables (features) of one object being classified. Should have data type CV_32F.
     * @return automatically generated
     */
    public float predict(Mat samples) {
        return predict_2(nativeObj, samples.nativeObj);
    }


    //
    // C++:  int cv::ml::LogisticRegression::getIterations()
    //

    /**
     * SEE: setIterations
     * @return automatically generated
     */
    public int getIterations() {
        return getIterations_0(nativeObj);
    }


    //
    // C++:  int cv::ml::LogisticRegression::getMiniBatchSize()
    //

    /**
     * SEE: setMiniBatchSize
     * @return automatically generated
     */
    public int getMiniBatchSize() {
        return getMiniBatchSize_0(nativeObj);
    }


    //
    // C++:  int cv::ml::LogisticRegression::getRegularization()
    //

    /**
     * SEE: setRegularization
     * @return automatically generated
     */
    public int getRegularization() {
        return getRegularization_0(nativeObj);
    }


    //
    // C++:  int cv::ml::LogisticRegression::getTrainMethod()
    //

    /**
     * SEE: setTrainMethod
     * @return automatically generated
     */
    public int getTrainMethod() {
        return getTrainMethod_0(nativeObj);
    }


    //
    // C++:  void cv::ml::LogisticRegression::setIterations(int val)
    //

    /**
     *  getIterations SEE: getIterations
     * @param val automatically generated
     */
    public void setIterations(int val) {
        setIterations_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::LogisticRegression::setLearningRate(double val)
    //

    /**
     *  getLearningRate SEE: getLearningRate
     * @param val automatically generated
     */
    public void setLearningRate(double val) {
        setLearningRate_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::LogisticRegression::setMiniBatchSize(int val)
    //

    /**
     *  getMiniBatchSize SEE: getMiniBatchSize
     * @param val automatically generated
     */
    public void setMiniBatchSize(int val) {
        setMiniBatchSize_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::LogisticRegression::setRegularization(int val)
    //

    /**
     *  getRegularization SEE: getRegularization
     * @param val automatically generated
     */
    public void setRegularization(int val) {
        setRegularization_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::LogisticRegression::setTermCriteria(TermCriteria val)
    //

    /**
     *  getTermCriteria SEE: getTermCriteria
     * @param val automatically generated
     */
    public void setTermCriteria(TermCriteria val) {
        setTermCriteria_0(nativeObj, val.type, val.maxCount, val.epsilon);
    }


    //
    // C++:  void cv::ml::LogisticRegression::setTrainMethod(int val)
    //

    /**
     *  getTrainMethod SEE: getTrainMethod
     * @param val automatically generated
     */
    public void setTrainMethod(int val) {
        setTrainMethod_0(nativeObj, val);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::ml::LogisticRegression::get_learnt_thetas()
    private static native long get_learnt_thetas_0(long nativeObj);

    // C++: static Ptr_LogisticRegression cv::ml::LogisticRegression::create()
    private static native long create_0();

    // C++: static Ptr_LogisticRegression cv::ml::LogisticRegression::load(String filepath, String nodeName = String())
    private static native long load_0(String filepath, String nodeName);
    private static native long load_1(String filepath);

    // C++:  TermCriteria cv::ml::LogisticRegression::getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  double cv::ml::LogisticRegression::getLearningRate()
    private static native double getLearningRate_0(long nativeObj);

    // C++:  float cv::ml::LogisticRegression::predict(Mat samples, Mat& results = Mat(), int flags = 0)
    private static native float predict_0(long nativeObj, long samples_nativeObj, long results_nativeObj, int flags);
    private static native float predict_1(long nativeObj, long samples_nativeObj, long results_nativeObj);
    private static native float predict_2(long nativeObj, long samples_nativeObj);

    // C++:  int cv::ml::LogisticRegression::getIterations()
    private static native int getIterations_0(long nativeObj);

    // C++:  int cv::ml::LogisticRegression::getMiniBatchSize()
    private static native int getMiniBatchSize_0(long nativeObj);

    // C++:  int cv::ml::LogisticRegression::getRegularization()
    private static native int getRegularization_0(long nativeObj);

    // C++:  int cv::ml::LogisticRegression::getTrainMethod()
    private static native int getTrainMethod_0(long nativeObj);

    // C++:  void cv::ml::LogisticRegression::setIterations(int val)
    private static native void setIterations_0(long nativeObj, int val);

    // C++:  void cv::ml::LogisticRegression::setLearningRate(double val)
    private static native void setLearningRate_0(long nativeObj, double val);

    // C++:  void cv::ml::LogisticRegression::setMiniBatchSize(int val)
    private static native void setMiniBatchSize_0(long nativeObj, int val);

    // C++:  void cv::ml::LogisticRegression::setRegularization(int val)
    private static native void setRegularization_0(long nativeObj, int val);

    // C++:  void cv::ml::LogisticRegression::setTermCriteria(TermCriteria val)
    private static native void setTermCriteria_0(long nativeObj, int val_type, int val_maxCount, double val_epsilon);

    // C++:  void cv::ml::LogisticRegression::setTrainMethod(int val)
    private static native void setTrainMethod_0(long nativeObj, int val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
