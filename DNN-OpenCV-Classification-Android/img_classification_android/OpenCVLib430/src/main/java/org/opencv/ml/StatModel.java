//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.ml.TrainData;

// C++: class StatModel
/**
 * Base class for statistical models in OpenCV ML.
 */
public class StatModel extends Algorithm {

    protected StatModel(long addr) { super(addr); }

    // internal usage only
    public static StatModel __fromPtr__(long addr) { return new StatModel(addr); }

    // C++: enum Flags
    public static final int
            UPDATE_MODEL = 1,
            RAW_OUTPUT = 1,
            COMPRESSED_INPUT = 2,
            PREPROCESSED_INPUT = 4;


    //
    // C++:  bool cv::ml::StatModel::empty()
    //

    public boolean empty() {
        return empty_0(nativeObj);
    }


    //
    // C++:  bool cv::ml::StatModel::isClassifier()
    //

    /**
     * Returns true if the model is classifier
     * @return automatically generated
     */
    public boolean isClassifier() {
        return isClassifier_0(nativeObj);
    }


    //
    // C++:  bool cv::ml::StatModel::isTrained()
    //

    /**
     * Returns true if the model is trained
     * @return automatically generated
     */
    public boolean isTrained() {
        return isTrained_0(nativeObj);
    }


    //
    // C++:  bool cv::ml::StatModel::train(Mat samples, int layout, Mat responses)
    //

    /**
     * Trains the statistical model
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     * @return automatically generated
     */
    public boolean train(Mat samples, int layout, Mat responses) {
        return train_0(nativeObj, samples.nativeObj, layout, responses.nativeObj);
    }


    //
    // C++:  bool cv::ml::StatModel::train(Ptr_TrainData trainData, int flags = 0)
    //

    /**
     * Trains the statistical model
     *
     *     @param trainData training data that can be loaded from file using TrainData::loadFromCSV or
     *         created with TrainData::create.
     *     @param flags optional flags, depending on the model. Some of the models can be updated with the
     *         new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).
     * @return automatically generated
     */
    public boolean train(TrainData trainData, int flags) {
        return train_1(nativeObj, trainData.getNativeObjAddr(), flags);
    }

    /**
     * Trains the statistical model
     *
     *     @param trainData training data that can be loaded from file using TrainData::loadFromCSV or
     *         created with TrainData::create.
     *         new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).
     * @return automatically generated
     */
    public boolean train(TrainData trainData) {
        return train_2(nativeObj, trainData.getNativeObjAddr());
    }


    //
    // C++:  float cv::ml::StatModel::calcError(Ptr_TrainData data, bool test, Mat& resp)
    //

    /**
     * Computes error on the training or test dataset
     *
     *     @param data the training data
     *     @param test if true, the error is computed over the test subset of the data, otherwise it's
     *         computed over the training subset of the data. Please note that if you loaded a completely
     *         different dataset to evaluate already trained classifier, you will probably want not to set
     *         the test subset at all with TrainData::setTrainTestSplitRatio and specify test=false, so
     *         that the error is computed for the whole new set. Yes, this sounds a bit confusing.
     *     @param resp the optional output responses.
     *
     *     The method uses StatModel::predict to compute the error. For regression models the error is
     *     computed as RMS, for classifiers - as a percent of missclassified samples (0%-100%).
     * @return automatically generated
     */
    public float calcError(TrainData data, boolean test, Mat resp) {
        return calcError_0(nativeObj, data.getNativeObjAddr(), test, resp.nativeObj);
    }


    //
    // C++:  float cv::ml::StatModel::predict(Mat samples, Mat& results = Mat(), int flags = 0)
    //

    /**
     * Predicts response(s) for the provided sample(s)
     *
     *     @param samples The input samples, floating-point matrix
     *     @param results The optional output matrix of results.
     *     @param flags The optional flags, model-dependent. See cv::ml::StatModel::Flags.
     * @return automatically generated
     */
    public float predict(Mat samples, Mat results, int flags) {
        return predict_0(nativeObj, samples.nativeObj, results.nativeObj, flags);
    }

    /**
     * Predicts response(s) for the provided sample(s)
     *
     *     @param samples The input samples, floating-point matrix
     *     @param results The optional output matrix of results.
     * @return automatically generated
     */
    public float predict(Mat samples, Mat results) {
        return predict_1(nativeObj, samples.nativeObj, results.nativeObj);
    }

    /**
     * Predicts response(s) for the provided sample(s)
     *
     *     @param samples The input samples, floating-point matrix
     * @return automatically generated
     */
    public float predict(Mat samples) {
        return predict_2(nativeObj, samples.nativeObj);
    }


    //
    // C++:  int cv::ml::StatModel::getVarCount()
    //

    /**
     * Returns the number of variables in training samples
     * @return automatically generated
     */
    public int getVarCount() {
        return getVarCount_0(nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  bool cv::ml::StatModel::empty()
    private static native boolean empty_0(long nativeObj);

    // C++:  bool cv::ml::StatModel::isClassifier()
    private static native boolean isClassifier_0(long nativeObj);

    // C++:  bool cv::ml::StatModel::isTrained()
    private static native boolean isTrained_0(long nativeObj);

    // C++:  bool cv::ml::StatModel::train(Mat samples, int layout, Mat responses)
    private static native boolean train_0(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj);

    // C++:  bool cv::ml::StatModel::train(Ptr_TrainData trainData, int flags = 0)
    private static native boolean train_1(long nativeObj, long trainData_nativeObj, int flags);
    private static native boolean train_2(long nativeObj, long trainData_nativeObj);

    // C++:  float cv::ml::StatModel::calcError(Ptr_TrainData data, bool test, Mat& resp)
    private static native float calcError_0(long nativeObj, long data_nativeObj, boolean test, long resp_nativeObj);

    // C++:  float cv::ml::StatModel::predict(Mat samples, Mat& results = Mat(), int flags = 0)
    private static native float predict_0(long nativeObj, long samples_nativeObj, long results_nativeObj, int flags);
    private static native float predict_1(long nativeObj, long samples_nativeObj, long results_nativeObj);
    private static native float predict_2(long nativeObj, long samples_nativeObj);

    // C++:  int cv::ml::StatModel::getVarCount()
    private static native int getVarCount_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
