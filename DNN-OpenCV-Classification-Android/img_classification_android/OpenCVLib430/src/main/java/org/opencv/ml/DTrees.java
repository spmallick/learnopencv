//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.ml.DTrees;
import org.opencv.ml.StatModel;

// C++: class DTrees
/**
 * The class represents a single decision tree or a collection of decision trees.
 *
 * The current public interface of the class allows user to train only a single decision tree, however
 * the class is capable of storing multiple decision trees and using them for prediction (by summing
 * responses or using a voting schemes), and the derived from DTrees classes (such as RTrees and Boost)
 * use this capability to implement decision tree ensembles.
 *
 * SEE: REF: ml_intro_trees
 */
public class DTrees extends StatModel {

    protected DTrees(long addr) { super(addr); }

    // internal usage only
    public static DTrees __fromPtr__(long addr) { return new DTrees(addr); }

    // C++: enum Flags
    public static final int
            PREDICT_AUTO = 0,
            PREDICT_SUM = (1<<8),
            PREDICT_MAX_VOTE = (2<<8),
            PREDICT_MASK = (3<<8);


    //
    // C++:  Mat cv::ml::DTrees::getPriors()
    //

    /**
     * SEE: setPriors
     * @return automatically generated
     */
    public Mat getPriors() {
        return new Mat(getPriors_0(nativeObj));
    }


    //
    // C++: static Ptr_DTrees cv::ml::DTrees::create()
    //

    /**
     * Creates the empty model
     *
     *     The static method creates empty decision tree with the specified parameters. It should be then
     *     trained using train method (see StatModel::train). Alternatively, you can load the model from
     *     file using Algorithm::load&lt;DTrees&gt;(filename).
     * @return automatically generated
     */
    public static DTrees create() {
        return DTrees.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_DTrees cv::ml::DTrees::load(String filepath, String nodeName = String())
    //

    /**
     * Loads and creates a serialized DTrees from a file
     *
     * Use DTree::save to serialize and store an DTree to disk.
     * Load the DTree from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized DTree
     * @param nodeName name of node containing the classifier
     * @return automatically generated
     */
    public static DTrees load(String filepath, String nodeName) {
        return DTrees.__fromPtr__(load_0(filepath, nodeName));
    }

    /**
     * Loads and creates a serialized DTrees from a file
     *
     * Use DTree::save to serialize and store an DTree to disk.
     * Load the DTree from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized DTree
     * @return automatically generated
     */
    public static DTrees load(String filepath) {
        return DTrees.__fromPtr__(load_1(filepath));
    }


    //
    // C++:  bool cv::ml::DTrees::getTruncatePrunedTree()
    //

    /**
     * SEE: setTruncatePrunedTree
     * @return automatically generated
     */
    public boolean getTruncatePrunedTree() {
        return getTruncatePrunedTree_0(nativeObj);
    }


    //
    // C++:  bool cv::ml::DTrees::getUse1SERule()
    //

    /**
     * SEE: setUse1SERule
     * @return automatically generated
     */
    public boolean getUse1SERule() {
        return getUse1SERule_0(nativeObj);
    }


    //
    // C++:  bool cv::ml::DTrees::getUseSurrogates()
    //

    /**
     * SEE: setUseSurrogates
     * @return automatically generated
     */
    public boolean getUseSurrogates() {
        return getUseSurrogates_0(nativeObj);
    }


    //
    // C++:  float cv::ml::DTrees::getRegressionAccuracy()
    //

    /**
     * SEE: setRegressionAccuracy
     * @return automatically generated
     */
    public float getRegressionAccuracy() {
        return getRegressionAccuracy_0(nativeObj);
    }


    //
    // C++:  int cv::ml::DTrees::getCVFolds()
    //

    /**
     * SEE: setCVFolds
     * @return automatically generated
     */
    public int getCVFolds() {
        return getCVFolds_0(nativeObj);
    }


    //
    // C++:  int cv::ml::DTrees::getMaxCategories()
    //

    /**
     * SEE: setMaxCategories
     * @return automatically generated
     */
    public int getMaxCategories() {
        return getMaxCategories_0(nativeObj);
    }


    //
    // C++:  int cv::ml::DTrees::getMaxDepth()
    //

    /**
     * SEE: setMaxDepth
     * @return automatically generated
     */
    public int getMaxDepth() {
        return getMaxDepth_0(nativeObj);
    }


    //
    // C++:  int cv::ml::DTrees::getMinSampleCount()
    //

    /**
     * SEE: setMinSampleCount
     * @return automatically generated
     */
    public int getMinSampleCount() {
        return getMinSampleCount_0(nativeObj);
    }


    //
    // C++:  void cv::ml::DTrees::setCVFolds(int val)
    //

    /**
     *  getCVFolds SEE: getCVFolds
     * @param val automatically generated
     */
    public void setCVFolds(int val) {
        setCVFolds_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::DTrees::setMaxCategories(int val)
    //

    /**
     *  getMaxCategories SEE: getMaxCategories
     * @param val automatically generated
     */
    public void setMaxCategories(int val) {
        setMaxCategories_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::DTrees::setMaxDepth(int val)
    //

    /**
     *  getMaxDepth SEE: getMaxDepth
     * @param val automatically generated
     */
    public void setMaxDepth(int val) {
        setMaxDepth_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::DTrees::setMinSampleCount(int val)
    //

    /**
     *  getMinSampleCount SEE: getMinSampleCount
     * @param val automatically generated
     */
    public void setMinSampleCount(int val) {
        setMinSampleCount_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::DTrees::setPriors(Mat val)
    //

    /**
     *  getPriors SEE: getPriors
     * @param val automatically generated
     */
    public void setPriors(Mat val) {
        setPriors_0(nativeObj, val.nativeObj);
    }


    //
    // C++:  void cv::ml::DTrees::setRegressionAccuracy(float val)
    //

    /**
     *  getRegressionAccuracy SEE: getRegressionAccuracy
     * @param val automatically generated
     */
    public void setRegressionAccuracy(float val) {
        setRegressionAccuracy_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::DTrees::setTruncatePrunedTree(bool val)
    //

    /**
     *  getTruncatePrunedTree SEE: getTruncatePrunedTree
     * @param val automatically generated
     */
    public void setTruncatePrunedTree(boolean val) {
        setTruncatePrunedTree_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::DTrees::setUse1SERule(bool val)
    //

    /**
     *  getUse1SERule SEE: getUse1SERule
     * @param val automatically generated
     */
    public void setUse1SERule(boolean val) {
        setUse1SERule_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::DTrees::setUseSurrogates(bool val)
    //

    /**
     *  getUseSurrogates SEE: getUseSurrogates
     * @param val automatically generated
     */
    public void setUseSurrogates(boolean val) {
        setUseSurrogates_0(nativeObj, val);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::ml::DTrees::getPriors()
    private static native long getPriors_0(long nativeObj);

    // C++: static Ptr_DTrees cv::ml::DTrees::create()
    private static native long create_0();

    // C++: static Ptr_DTrees cv::ml::DTrees::load(String filepath, String nodeName = String())
    private static native long load_0(String filepath, String nodeName);
    private static native long load_1(String filepath);

    // C++:  bool cv::ml::DTrees::getTruncatePrunedTree()
    private static native boolean getTruncatePrunedTree_0(long nativeObj);

    // C++:  bool cv::ml::DTrees::getUse1SERule()
    private static native boolean getUse1SERule_0(long nativeObj);

    // C++:  bool cv::ml::DTrees::getUseSurrogates()
    private static native boolean getUseSurrogates_0(long nativeObj);

    // C++:  float cv::ml::DTrees::getRegressionAccuracy()
    private static native float getRegressionAccuracy_0(long nativeObj);

    // C++:  int cv::ml::DTrees::getCVFolds()
    private static native int getCVFolds_0(long nativeObj);

    // C++:  int cv::ml::DTrees::getMaxCategories()
    private static native int getMaxCategories_0(long nativeObj);

    // C++:  int cv::ml::DTrees::getMaxDepth()
    private static native int getMaxDepth_0(long nativeObj);

    // C++:  int cv::ml::DTrees::getMinSampleCount()
    private static native int getMinSampleCount_0(long nativeObj);

    // C++:  void cv::ml::DTrees::setCVFolds(int val)
    private static native void setCVFolds_0(long nativeObj, int val);

    // C++:  void cv::ml::DTrees::setMaxCategories(int val)
    private static native void setMaxCategories_0(long nativeObj, int val);

    // C++:  void cv::ml::DTrees::setMaxDepth(int val)
    private static native void setMaxDepth_0(long nativeObj, int val);

    // C++:  void cv::ml::DTrees::setMinSampleCount(int val)
    private static native void setMinSampleCount_0(long nativeObj, int val);

    // C++:  void cv::ml::DTrees::setPriors(Mat val)
    private static native void setPriors_0(long nativeObj, long val_nativeObj);

    // C++:  void cv::ml::DTrees::setRegressionAccuracy(float val)
    private static native void setRegressionAccuracy_0(long nativeObj, float val);

    // C++:  void cv::ml::DTrees::setTruncatePrunedTree(bool val)
    private static native void setTruncatePrunedTree_0(long nativeObj, boolean val);

    // C++:  void cv::ml::DTrees::setUse1SERule(bool val)
    private static native void setUse1SERule_0(long nativeObj, boolean val);

    // C++:  void cv::ml::DTrees::setUseSurrogates(bool val)
    private static native void setUseSurrogates_0(long nativeObj, boolean val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
