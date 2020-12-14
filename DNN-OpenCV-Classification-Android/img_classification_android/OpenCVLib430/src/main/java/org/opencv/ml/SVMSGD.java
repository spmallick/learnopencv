//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.SVMSGD;
import org.opencv.ml.StatModel;

// C++: class SVMSGD
/**
 * *************************************************************************************\
 * Stochastic Gradient Descent SVM Classifier                      *
 * \***************************************************************************************
 */
public class SVMSGD extends StatModel {

    protected SVMSGD(long addr) { super(addr); }

    // internal usage only
    public static SVMSGD __fromPtr__(long addr) { return new SVMSGD(addr); }

    // C++: enum SvmsgdType
    public static final int
            SGD = 0,
            ASGD = 1;


    // C++: enum MarginType
    public static final int
            SOFT_MARGIN = 0,
            HARD_MARGIN = 1;


    //
    // C++:  Mat cv::ml::SVMSGD::getWeights()
    //

    /**
     * @return the weights of the trained model (decision function f(x) = weights * x + shift).
     */
    public Mat getWeights() {
        return new Mat(getWeights_0(nativeObj));
    }


    //
    // C++: static Ptr_SVMSGD cv::ml::SVMSGD::create()
    //

    /**
     * Creates empty model.
     * Use StatModel::train to train the model. Since %SVMSGD has several parameters, you may want to
     * find the best parameters for your problem or use setOptimalParameters() to set some default parameters.
     * @return automatically generated
     */
    public static SVMSGD create() {
        return SVMSGD.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_SVMSGD cv::ml::SVMSGD::load(String filepath, String nodeName = String())
    //

    /**
     * Loads and creates a serialized SVMSGD from a file
     *
     * Use SVMSGD::save to serialize and store an SVMSGD to disk.
     * Load the SVMSGD from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized SVMSGD
     * @param nodeName name of node containing the classifier
     * @return automatically generated
     */
    public static SVMSGD load(String filepath, String nodeName) {
        return SVMSGD.__fromPtr__(load_0(filepath, nodeName));
    }

    /**
     * Loads and creates a serialized SVMSGD from a file
     *
     * Use SVMSGD::save to serialize and store an SVMSGD to disk.
     * Load the SVMSGD from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized SVMSGD
     * @return automatically generated
     */
    public static SVMSGD load(String filepath) {
        return SVMSGD.__fromPtr__(load_1(filepath));
    }


    //
    // C++:  TermCriteria cv::ml::SVMSGD::getTermCriteria()
    //

    /**
     * SEE: setTermCriteria
     * @return automatically generated
     */
    public TermCriteria getTermCriteria() {
        return new TermCriteria(getTermCriteria_0(nativeObj));
    }


    //
    // C++:  float cv::ml::SVMSGD::getInitialStepSize()
    //

    /**
     * SEE: setInitialStepSize
     * @return automatically generated
     */
    public float getInitialStepSize() {
        return getInitialStepSize_0(nativeObj);
    }


    //
    // C++:  float cv::ml::SVMSGD::getMarginRegularization()
    //

    /**
     * SEE: setMarginRegularization
     * @return automatically generated
     */
    public float getMarginRegularization() {
        return getMarginRegularization_0(nativeObj);
    }


    //
    // C++:  float cv::ml::SVMSGD::getShift()
    //

    /**
     * @return the shift of the trained model (decision function f(x) = weights * x + shift).
     */
    public float getShift() {
        return getShift_0(nativeObj);
    }


    //
    // C++:  float cv::ml::SVMSGD::getStepDecreasingPower()
    //

    /**
     * SEE: setStepDecreasingPower
     * @return automatically generated
     */
    public float getStepDecreasingPower() {
        return getStepDecreasingPower_0(nativeObj);
    }


    //
    // C++:  int cv::ml::SVMSGD::getMarginType()
    //

    /**
     * SEE: setMarginType
     * @return automatically generated
     */
    public int getMarginType() {
        return getMarginType_0(nativeObj);
    }


    //
    // C++:  int cv::ml::SVMSGD::getSvmsgdType()
    //

    /**
     * SEE: setSvmsgdType
     * @return automatically generated
     */
    public int getSvmsgdType() {
        return getSvmsgdType_0(nativeObj);
    }


    //
    // C++:  void cv::ml::SVMSGD::setInitialStepSize(float InitialStepSize)
    //

    /**
     *  getInitialStepSize SEE: getInitialStepSize
     * @param InitialStepSize automatically generated
     */
    public void setInitialStepSize(float InitialStepSize) {
        setInitialStepSize_0(nativeObj, InitialStepSize);
    }


    //
    // C++:  void cv::ml::SVMSGD::setMarginRegularization(float marginRegularization)
    //

    /**
     *  getMarginRegularization SEE: getMarginRegularization
     * @param marginRegularization automatically generated
     */
    public void setMarginRegularization(float marginRegularization) {
        setMarginRegularization_0(nativeObj, marginRegularization);
    }


    //
    // C++:  void cv::ml::SVMSGD::setMarginType(int marginType)
    //

    /**
     *  getMarginType SEE: getMarginType
     * @param marginType automatically generated
     */
    public void setMarginType(int marginType) {
        setMarginType_0(nativeObj, marginType);
    }


    //
    // C++:  void cv::ml::SVMSGD::setOptimalParameters(int svmsgdType = SVMSGD::ASGD, int marginType = SVMSGD::SOFT_MARGIN)
    //

    /**
     * Function sets optimal parameters values for chosen SVM SGD model.
     * @param svmsgdType is the type of SVMSGD classifier.
     * @param marginType is the type of margin constraint.
     */
    public void setOptimalParameters(int svmsgdType, int marginType) {
        setOptimalParameters_0(nativeObj, svmsgdType, marginType);
    }

    /**
     * Function sets optimal parameters values for chosen SVM SGD model.
     * @param svmsgdType is the type of SVMSGD classifier.
     */
    public void setOptimalParameters(int svmsgdType) {
        setOptimalParameters_1(nativeObj, svmsgdType);
    }

    /**
     * Function sets optimal parameters values for chosen SVM SGD model.
     */
    public void setOptimalParameters() {
        setOptimalParameters_2(nativeObj);
    }


    //
    // C++:  void cv::ml::SVMSGD::setStepDecreasingPower(float stepDecreasingPower)
    //

    /**
     *  getStepDecreasingPower SEE: getStepDecreasingPower
     * @param stepDecreasingPower automatically generated
     */
    public void setStepDecreasingPower(float stepDecreasingPower) {
        setStepDecreasingPower_0(nativeObj, stepDecreasingPower);
    }


    //
    // C++:  void cv::ml::SVMSGD::setSvmsgdType(int svmsgdType)
    //

    /**
     *  getSvmsgdType SEE: getSvmsgdType
     * @param svmsgdType automatically generated
     */
    public void setSvmsgdType(int svmsgdType) {
        setSvmsgdType_0(nativeObj, svmsgdType);
    }


    //
    // C++:  void cv::ml::SVMSGD::setTermCriteria(TermCriteria val)
    //

    /**
     *  getTermCriteria SEE: getTermCriteria
     * @param val automatically generated
     */
    public void setTermCriteria(TermCriteria val) {
        setTermCriteria_0(nativeObj, val.type, val.maxCount, val.epsilon);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::ml::SVMSGD::getWeights()
    private static native long getWeights_0(long nativeObj);

    // C++: static Ptr_SVMSGD cv::ml::SVMSGD::create()
    private static native long create_0();

    // C++: static Ptr_SVMSGD cv::ml::SVMSGD::load(String filepath, String nodeName = String())
    private static native long load_0(String filepath, String nodeName);
    private static native long load_1(String filepath);

    // C++:  TermCriteria cv::ml::SVMSGD::getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  float cv::ml::SVMSGD::getInitialStepSize()
    private static native float getInitialStepSize_0(long nativeObj);

    // C++:  float cv::ml::SVMSGD::getMarginRegularization()
    private static native float getMarginRegularization_0(long nativeObj);

    // C++:  float cv::ml::SVMSGD::getShift()
    private static native float getShift_0(long nativeObj);

    // C++:  float cv::ml::SVMSGD::getStepDecreasingPower()
    private static native float getStepDecreasingPower_0(long nativeObj);

    // C++:  int cv::ml::SVMSGD::getMarginType()
    private static native int getMarginType_0(long nativeObj);

    // C++:  int cv::ml::SVMSGD::getSvmsgdType()
    private static native int getSvmsgdType_0(long nativeObj);

    // C++:  void cv::ml::SVMSGD::setInitialStepSize(float InitialStepSize)
    private static native void setInitialStepSize_0(long nativeObj, float InitialStepSize);

    // C++:  void cv::ml::SVMSGD::setMarginRegularization(float marginRegularization)
    private static native void setMarginRegularization_0(long nativeObj, float marginRegularization);

    // C++:  void cv::ml::SVMSGD::setMarginType(int marginType)
    private static native void setMarginType_0(long nativeObj, int marginType);

    // C++:  void cv::ml::SVMSGD::setOptimalParameters(int svmsgdType = SVMSGD::ASGD, int marginType = SVMSGD::SOFT_MARGIN)
    private static native void setOptimalParameters_0(long nativeObj, int svmsgdType, int marginType);
    private static native void setOptimalParameters_1(long nativeObj, int svmsgdType);
    private static native void setOptimalParameters_2(long nativeObj);

    // C++:  void cv::ml::SVMSGD::setStepDecreasingPower(float stepDecreasingPower)
    private static native void setStepDecreasingPower_0(long nativeObj, float stepDecreasingPower);

    // C++:  void cv::ml::SVMSGD::setSvmsgdType(int svmsgdType)
    private static native void setSvmsgdType_0(long nativeObj, int svmsgdType);

    // C++:  void cv::ml::SVMSGD::setTermCriteria(TermCriteria val)
    private static native void setTermCriteria_0(long nativeObj, int val_type, int val_maxCount, double val_epsilon);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
