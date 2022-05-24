//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.DTrees;
import org.opencv.ml.RTrees;

// C++: class RTrees
/**
 * The class implements the random forest predictor.
 *
 * SEE: REF: ml_intro_rtrees
 */
public class RTrees extends DTrees {

    protected RTrees(long addr) { super(addr); }

    // internal usage only
    public static RTrees __fromPtr__(long addr) { return new RTrees(addr); }

    //
    // C++:  Mat cv::ml::RTrees::getVarImportance()
    //

    /**
     * Returns the variable importance array.
     *     The method returns the variable importance vector, computed at the training stage when
     *     CalculateVarImportance is set to true. If this flag was set to false, the empty matrix is
     *     returned.
     * @return automatically generated
     */
    public Mat getVarImportance() {
        return new Mat(getVarImportance_0(nativeObj));
    }


    //
    // C++: static Ptr_RTrees cv::ml::RTrees::create()
    //

    /**
     * Creates the empty model.
     *     Use StatModel::train to train the model, StatModel::train to create and train the model,
     *     Algorithm::load to load the pre-trained model.
     * @return automatically generated
     */
    public static RTrees create() {
        return RTrees.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_RTrees cv::ml::RTrees::load(String filepath, String nodeName = String())
    //

    /**
     * Loads and creates a serialized RTree from a file
     *
     * Use RTree::save to serialize and store an RTree to disk.
     * Load the RTree from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized RTree
     * @param nodeName name of node containing the classifier
     * @return automatically generated
     */
    public static RTrees load(String filepath, String nodeName) {
        return RTrees.__fromPtr__(load_0(filepath, nodeName));
    }

    /**
     * Loads and creates a serialized RTree from a file
     *
     * Use RTree::save to serialize and store an RTree to disk.
     * Load the RTree from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized RTree
     * @return automatically generated
     */
    public static RTrees load(String filepath) {
        return RTrees.__fromPtr__(load_1(filepath));
    }


    //
    // C++:  TermCriteria cv::ml::RTrees::getTermCriteria()
    //

    /**
     * SEE: setTermCriteria
     * @return automatically generated
     */
    public TermCriteria getTermCriteria() {
        return new TermCriteria(getTermCriteria_0(nativeObj));
    }


    //
    // C++:  bool cv::ml::RTrees::getCalculateVarImportance()
    //

    /**
     * SEE: setCalculateVarImportance
     * @return automatically generated
     */
    public boolean getCalculateVarImportance() {
        return getCalculateVarImportance_0(nativeObj);
    }


    //
    // C++:  int cv::ml::RTrees::getActiveVarCount()
    //

    /**
     * SEE: setActiveVarCount
     * @return automatically generated
     */
    public int getActiveVarCount() {
        return getActiveVarCount_0(nativeObj);
    }


    //
    // C++:  void cv::ml::RTrees::getVotes(Mat samples, Mat& results, int flags)
    //

    /**
     * Returns the result of each individual tree in the forest.
     *     In case the model is a regression problem, the method will return each of the trees'
     *     results for each of the sample cases. If the model is a classifier, it will return
     *     a Mat with samples + 1 rows, where the first row gives the class number and the
     *     following rows return the votes each class had for each sample.
     *         @param samples Array containing the samples for which votes will be calculated.
     *         @param results Array where the result of the calculation will be written.
     *         @param flags Flags for defining the type of RTrees.
     */
    public void getVotes(Mat samples, Mat results, int flags) {
        getVotes_0(nativeObj, samples.nativeObj, results.nativeObj, flags);
    }


    //
    // C++:  void cv::ml::RTrees::setActiveVarCount(int val)
    //

    /**
     *  getActiveVarCount SEE: getActiveVarCount
     * @param val automatically generated
     */
    public void setActiveVarCount(int val) {
        setActiveVarCount_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::RTrees::setCalculateVarImportance(bool val)
    //

    /**
     *  getCalculateVarImportance SEE: getCalculateVarImportance
     * @param val automatically generated
     */
    public void setCalculateVarImportance(boolean val) {
        setCalculateVarImportance_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::RTrees::setTermCriteria(TermCriteria val)
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



    // C++:  Mat cv::ml::RTrees::getVarImportance()
    private static native long getVarImportance_0(long nativeObj);

    // C++: static Ptr_RTrees cv::ml::RTrees::create()
    private static native long create_0();

    // C++: static Ptr_RTrees cv::ml::RTrees::load(String filepath, String nodeName = String())
    private static native long load_0(String filepath, String nodeName);
    private static native long load_1(String filepath);

    // C++:  TermCriteria cv::ml::RTrees::getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  bool cv::ml::RTrees::getCalculateVarImportance()
    private static native boolean getCalculateVarImportance_0(long nativeObj);

    // C++:  int cv::ml::RTrees::getActiveVarCount()
    private static native int getActiveVarCount_0(long nativeObj);

    // C++:  void cv::ml::RTrees::getVotes(Mat samples, Mat& results, int flags)
    private static native void getVotes_0(long nativeObj, long samples_nativeObj, long results_nativeObj, int flags);

    // C++:  void cv::ml::RTrees::setActiveVarCount(int val)
    private static native void setActiveVarCount_0(long nativeObj, int val);

    // C++:  void cv::ml::RTrees::setCalculateVarImportance(bool val)
    private static native void setCalculateVarImportance_0(long nativeObj, boolean val);

    // C++:  void cv::ml::RTrees::setTermCriteria(TermCriteria val)
    private static native void setTermCriteria_0(long nativeObj, int val_type, int val_maxCount, double val_epsilon);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
