//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.StatModel;

// C++: class ANN_MLP
/**
 * Artificial Neural Networks - Multi-Layer Perceptrons.
 *
 * Unlike many other models in ML that are constructed and trained at once, in the MLP model these
 * steps are separated. First, a network with the specified topology is created using the non-default
 * constructor or the method ANN_MLP::create. All the weights are set to zeros. Then, the network is
 * trained using a set of input and output vectors. The training procedure can be repeated more than
 * once, that is, the weights can be adjusted based on the new training data.
 *
 * Additional flags for StatModel::train are available: ANN_MLP::TrainFlags.
 *
 * SEE: REF: ml_intro_ann
 */
public class ANN_MLP extends StatModel {

    protected ANN_MLP(long addr) { super(addr); }

    // internal usage only
    public static ANN_MLP __fromPtr__(long addr) { return new ANN_MLP(addr); }

    // C++: enum TrainingMethods
    public static final int
            BACKPROP = 0,
            RPROP = 1,
            ANNEAL = 2;


    // C++: enum TrainFlags
    public static final int
            UPDATE_WEIGHTS = 1,
            NO_INPUT_SCALE = 2,
            NO_OUTPUT_SCALE = 4;


    // C++: enum ActivationFunctions
    public static final int
            IDENTITY = 0,
            SIGMOID_SYM = 1,
            GAUSSIAN = 2,
            RELU = 3,
            LEAKYRELU = 4;


    //
    // C++:  Mat cv::ml::ANN_MLP::getLayerSizes()
    //

    /**
     * Integer vector specifying the number of neurons in each layer including the input and output layers.
     *     The very first element specifies the number of elements in the input layer.
     *     The last element - number of elements in the output layer.
     * SEE: setLayerSizes
     * @return automatically generated
     */
    public Mat getLayerSizes() {
        return new Mat(getLayerSizes_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::ANN_MLP::getWeights(int layerIdx)
    //

    public Mat getWeights(int layerIdx) {
        return new Mat(getWeights_0(nativeObj, layerIdx));
    }


    //
    // C++: static Ptr_ANN_MLP cv::ml::ANN_MLP::create()
    //

    /**
     * Creates empty model
     *
     *     Use StatModel::train to train the model, Algorithm::load&lt;ANN_MLP&gt;(filename) to load the pre-trained model.
     *     Note that the train method has optional flags: ANN_MLP::TrainFlags.
     * @return automatically generated
     */
    public static ANN_MLP create() {
        return ANN_MLP.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_ANN_MLP cv::ml::ANN_MLP::load(String filepath)
    //

    /**
     * Loads and creates a serialized ANN from a file
     *
     * Use ANN::save to serialize and store an ANN to disk.
     * Load the ANN from this file again, by calling this function with the path to the file.
     *
     * @param filepath path to serialized ANN
     * @return automatically generated
     */
    public static ANN_MLP load(String filepath) {
        return ANN_MLP.__fromPtr__(load_0(filepath));
    }


    //
    // C++:  TermCriteria cv::ml::ANN_MLP::getTermCriteria()
    //

    /**
     * SEE: setTermCriteria
     * @return automatically generated
     */
    public TermCriteria getTermCriteria() {
        return new TermCriteria(getTermCriteria_0(nativeObj));
    }


    //
    // C++:  double cv::ml::ANN_MLP::getAnnealCoolingRatio()
    //

    /**
     * SEE: setAnnealCoolingRatio
     * @return automatically generated
     */
    public double getAnnealCoolingRatio() {
        return getAnnealCoolingRatio_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getAnnealFinalT()
    //

    /**
     * SEE: setAnnealFinalT
     * @return automatically generated
     */
    public double getAnnealFinalT() {
        return getAnnealFinalT_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getAnnealInitialT()
    //

    /**
     * SEE: setAnnealInitialT
     * @return automatically generated
     */
    public double getAnnealInitialT() {
        return getAnnealInitialT_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getBackpropMomentumScale()
    //

    /**
     * SEE: setBackpropMomentumScale
     * @return automatically generated
     */
    public double getBackpropMomentumScale() {
        return getBackpropMomentumScale_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getBackpropWeightScale()
    //

    /**
     * SEE: setBackpropWeightScale
     * @return automatically generated
     */
    public double getBackpropWeightScale() {
        return getBackpropWeightScale_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getRpropDW0()
    //

    /**
     * SEE: setRpropDW0
     * @return automatically generated
     */
    public double getRpropDW0() {
        return getRpropDW0_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getRpropDWMax()
    //

    /**
     * SEE: setRpropDWMax
     * @return automatically generated
     */
    public double getRpropDWMax() {
        return getRpropDWMax_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getRpropDWMin()
    //

    /**
     * SEE: setRpropDWMin
     * @return automatically generated
     */
    public double getRpropDWMin() {
        return getRpropDWMin_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getRpropDWMinus()
    //

    /**
     * SEE: setRpropDWMinus
     * @return automatically generated
     */
    public double getRpropDWMinus() {
        return getRpropDWMinus_0(nativeObj);
    }


    //
    // C++:  double cv::ml::ANN_MLP::getRpropDWPlus()
    //

    /**
     * SEE: setRpropDWPlus
     * @return automatically generated
     */
    public double getRpropDWPlus() {
        return getRpropDWPlus_0(nativeObj);
    }


    //
    // C++:  int cv::ml::ANN_MLP::getAnnealItePerStep()
    //

    /**
     * SEE: setAnnealItePerStep
     * @return automatically generated
     */
    public int getAnnealItePerStep() {
        return getAnnealItePerStep_0(nativeObj);
    }


    //
    // C++:  int cv::ml::ANN_MLP::getTrainMethod()
    //

    /**
     * Returns current training method
     * @return automatically generated
     */
    public int getTrainMethod() {
        return getTrainMethod_0(nativeObj);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setActivationFunction(int type, double param1 = 0, double param2 = 0)
    //

    /**
     * Initialize the activation function for each neuron.
     *     Currently the default and the only fully supported activation function is ANN_MLP::SIGMOID_SYM.
     *     @param type The type of activation function. See ANN_MLP::ActivationFunctions.
     *     @param param1 The first parameter of the activation function, \(\alpha\). Default value is 0.
     *     @param param2 The second parameter of the activation function, \(\beta\). Default value is 0.
     */
    public void setActivationFunction(int type, double param1, double param2) {
        setActivationFunction_0(nativeObj, type, param1, param2);
    }

    /**
     * Initialize the activation function for each neuron.
     *     Currently the default and the only fully supported activation function is ANN_MLP::SIGMOID_SYM.
     *     @param type The type of activation function. See ANN_MLP::ActivationFunctions.
     *     @param param1 The first parameter of the activation function, \(\alpha\). Default value is 0.
     */
    public void setActivationFunction(int type, double param1) {
        setActivationFunction_1(nativeObj, type, param1);
    }

    /**
     * Initialize the activation function for each neuron.
     *     Currently the default and the only fully supported activation function is ANN_MLP::SIGMOID_SYM.
     *     @param type The type of activation function. See ANN_MLP::ActivationFunctions.
     */
    public void setActivationFunction(int type) {
        setActivationFunction_2(nativeObj, type);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setAnnealCoolingRatio(double val)
    //

    /**
     *  getAnnealCoolingRatio SEE: getAnnealCoolingRatio
     * @param val automatically generated
     */
    public void setAnnealCoolingRatio(double val) {
        setAnnealCoolingRatio_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setAnnealFinalT(double val)
    //

    /**
     *  getAnnealFinalT SEE: getAnnealFinalT
     * @param val automatically generated
     */
    public void setAnnealFinalT(double val) {
        setAnnealFinalT_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setAnnealInitialT(double val)
    //

    /**
     *  getAnnealInitialT SEE: getAnnealInitialT
     * @param val automatically generated
     */
    public void setAnnealInitialT(double val) {
        setAnnealInitialT_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setAnnealItePerStep(int val)
    //

    /**
     *  getAnnealItePerStep SEE: getAnnealItePerStep
     * @param val automatically generated
     */
    public void setAnnealItePerStep(int val) {
        setAnnealItePerStep_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setBackpropMomentumScale(double val)
    //

    /**
     *  getBackpropMomentumScale SEE: getBackpropMomentumScale
     * @param val automatically generated
     */
    public void setBackpropMomentumScale(double val) {
        setBackpropMomentumScale_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setBackpropWeightScale(double val)
    //

    /**
     *  getBackpropWeightScale SEE: getBackpropWeightScale
     * @param val automatically generated
     */
    public void setBackpropWeightScale(double val) {
        setBackpropWeightScale_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setLayerSizes(Mat _layer_sizes)
    //

    /**
     * Integer vector specifying the number of neurons in each layer including the input and output layers.
     *     The very first element specifies the number of elements in the input layer.
     *     The last element - number of elements in the output layer. Default value is empty Mat.
     * SEE: getLayerSizes
     * @param _layer_sizes automatically generated
     */
    public void setLayerSizes(Mat _layer_sizes) {
        setLayerSizes_0(nativeObj, _layer_sizes.nativeObj);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setRpropDW0(double val)
    //

    /**
     *  getRpropDW0 SEE: getRpropDW0
     * @param val automatically generated
     */
    public void setRpropDW0(double val) {
        setRpropDW0_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setRpropDWMax(double val)
    //

    /**
     *  getRpropDWMax SEE: getRpropDWMax
     * @param val automatically generated
     */
    public void setRpropDWMax(double val) {
        setRpropDWMax_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setRpropDWMin(double val)
    //

    /**
     *  getRpropDWMin SEE: getRpropDWMin
     * @param val automatically generated
     */
    public void setRpropDWMin(double val) {
        setRpropDWMin_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setRpropDWMinus(double val)
    //

    /**
     *  getRpropDWMinus SEE: getRpropDWMinus
     * @param val automatically generated
     */
    public void setRpropDWMinus(double val) {
        setRpropDWMinus_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setRpropDWPlus(double val)
    //

    /**
     *  getRpropDWPlus SEE: getRpropDWPlus
     * @param val automatically generated
     */
    public void setRpropDWPlus(double val) {
        setRpropDWPlus_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setTermCriteria(TermCriteria val)
    //

    /**
     *  getTermCriteria SEE: getTermCriteria
     * @param val automatically generated
     */
    public void setTermCriteria(TermCriteria val) {
        setTermCriteria_0(nativeObj, val.type, val.maxCount, val.epsilon);
    }


    //
    // C++:  void cv::ml::ANN_MLP::setTrainMethod(int method, double param1 = 0, double param2 = 0)
    //

    /**
     * Sets training method and common parameters.
     *     @param method Default value is ANN_MLP::RPROP. See ANN_MLP::TrainingMethods.
     *     @param param1 passed to setRpropDW0 for ANN_MLP::RPROP and to setBackpropWeightScale for ANN_MLP::BACKPROP and to initialT for ANN_MLP::ANNEAL.
     *     @param param2 passed to setRpropDWMin for ANN_MLP::RPROP and to setBackpropMomentumScale for ANN_MLP::BACKPROP and to finalT for ANN_MLP::ANNEAL.
     */
    public void setTrainMethod(int method, double param1, double param2) {
        setTrainMethod_0(nativeObj, method, param1, param2);
    }

    /**
     * Sets training method and common parameters.
     *     @param method Default value is ANN_MLP::RPROP. See ANN_MLP::TrainingMethods.
     *     @param param1 passed to setRpropDW0 for ANN_MLP::RPROP and to setBackpropWeightScale for ANN_MLP::BACKPROP and to initialT for ANN_MLP::ANNEAL.
     */
    public void setTrainMethod(int method, double param1) {
        setTrainMethod_1(nativeObj, method, param1);
    }

    /**
     * Sets training method and common parameters.
     *     @param method Default value is ANN_MLP::RPROP. See ANN_MLP::TrainingMethods.
     */
    public void setTrainMethod(int method) {
        setTrainMethod_2(nativeObj, method);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::ml::ANN_MLP::getLayerSizes()
    private static native long getLayerSizes_0(long nativeObj);

    // C++:  Mat cv::ml::ANN_MLP::getWeights(int layerIdx)
    private static native long getWeights_0(long nativeObj, int layerIdx);

    // C++: static Ptr_ANN_MLP cv::ml::ANN_MLP::create()
    private static native long create_0();

    // C++: static Ptr_ANN_MLP cv::ml::ANN_MLP::load(String filepath)
    private static native long load_0(String filepath);

    // C++:  TermCriteria cv::ml::ANN_MLP::getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getAnnealCoolingRatio()
    private static native double getAnnealCoolingRatio_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getAnnealFinalT()
    private static native double getAnnealFinalT_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getAnnealInitialT()
    private static native double getAnnealInitialT_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getBackpropMomentumScale()
    private static native double getBackpropMomentumScale_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getBackpropWeightScale()
    private static native double getBackpropWeightScale_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getRpropDW0()
    private static native double getRpropDW0_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getRpropDWMax()
    private static native double getRpropDWMax_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getRpropDWMin()
    private static native double getRpropDWMin_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getRpropDWMinus()
    private static native double getRpropDWMinus_0(long nativeObj);

    // C++:  double cv::ml::ANN_MLP::getRpropDWPlus()
    private static native double getRpropDWPlus_0(long nativeObj);

    // C++:  int cv::ml::ANN_MLP::getAnnealItePerStep()
    private static native int getAnnealItePerStep_0(long nativeObj);

    // C++:  int cv::ml::ANN_MLP::getTrainMethod()
    private static native int getTrainMethod_0(long nativeObj);

    // C++:  void cv::ml::ANN_MLP::setActivationFunction(int type, double param1 = 0, double param2 = 0)
    private static native void setActivationFunction_0(long nativeObj, int type, double param1, double param2);
    private static native void setActivationFunction_1(long nativeObj, int type, double param1);
    private static native void setActivationFunction_2(long nativeObj, int type);

    // C++:  void cv::ml::ANN_MLP::setAnnealCoolingRatio(double val)
    private static native void setAnnealCoolingRatio_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setAnnealFinalT(double val)
    private static native void setAnnealFinalT_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setAnnealInitialT(double val)
    private static native void setAnnealInitialT_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setAnnealItePerStep(int val)
    private static native void setAnnealItePerStep_0(long nativeObj, int val);

    // C++:  void cv::ml::ANN_MLP::setBackpropMomentumScale(double val)
    private static native void setBackpropMomentumScale_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setBackpropWeightScale(double val)
    private static native void setBackpropWeightScale_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setLayerSizes(Mat _layer_sizes)
    private static native void setLayerSizes_0(long nativeObj, long _layer_sizes_nativeObj);

    // C++:  void cv::ml::ANN_MLP::setRpropDW0(double val)
    private static native void setRpropDW0_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setRpropDWMax(double val)
    private static native void setRpropDWMax_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setRpropDWMin(double val)
    private static native void setRpropDWMin_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setRpropDWMinus(double val)
    private static native void setRpropDWMinus_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setRpropDWPlus(double val)
    private static native void setRpropDWPlus_0(long nativeObj, double val);

    // C++:  void cv::ml::ANN_MLP::setTermCriteria(TermCriteria val)
    private static native void setTermCriteria_0(long nativeObj, int val_type, int val_maxCount, double val_epsilon);

    // C++:  void cv::ml::ANN_MLP::setTrainMethod(int method, double param1 = 0, double param2 = 0)
    private static native void setTrainMethod_0(long nativeObj, int method, double param1, double param2);
    private static native void setTrainMethod_1(long nativeObj, int method, double param1);
    private static native void setTrainMethod_2(long nativeObj, int method);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
