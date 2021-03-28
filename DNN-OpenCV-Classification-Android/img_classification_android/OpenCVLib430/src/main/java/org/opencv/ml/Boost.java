//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.ml.Boost;
import org.opencv.ml.DTrees;

// C++: class Boost
/**
 * Boosted tree classifier derived from DTrees
 *
 * SEE: REF: ml_intro_boost
 */
public class Boost extends DTrees {

    protected Boost(long addr) { super(addr); }

    // internal usage only
    public static Boost __fromPtr__(long addr) { return new Boost(addr); }

    // C++: enum Types
    public static final int
            DISCRETE = 0,
            REAL = 1,
            LOGIT = 2,
            GENTLE = 3;


    //
    // C++: static Ptr_Boost cv::ml::Boost::create()
    //

    /**
     * Creates the empty model.
     * Use StatModel::train to train the model, Algorithm::load&lt;Boost&gt;(filename) to load the pre-trained model.
     * @return automatically generated
     */
    public static Boost create() {
        return Boost.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_Boost cv::ml::Boost::load(String filepath, String nodeName = String())
    //

    /**
     * Loads and creates a serialized Boost from a file
     *
     * Use Boost::save to serialize and store an RTree to disk.
     * Load the Boost from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized Boost
     * @param nodeName name of node containing the classifier
     * @return automatically generated
     */
    public static Boost load(String filepath, String nodeName) {
        return Boost.__fromPtr__(load_0(filepath, nodeName));
    }

    /**
     * Loads and creates a serialized Boost from a file
     *
     * Use Boost::save to serialize and store an RTree to disk.
     * Load the Boost from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized Boost
     * @return automatically generated
     */
    public static Boost load(String filepath) {
        return Boost.__fromPtr__(load_1(filepath));
    }


    //
    // C++:  double cv::ml::Boost::getWeightTrimRate()
    //

    /**
     * SEE: setWeightTrimRate
     * @return automatically generated
     */
    public double getWeightTrimRate() {
        return getWeightTrimRate_0(nativeObj);
    }


    //
    // C++:  int cv::ml::Boost::getBoostType()
    //

    /**
     * SEE: setBoostType
     * @return automatically generated
     */
    public int getBoostType() {
        return getBoostType_0(nativeObj);
    }


    //
    // C++:  int cv::ml::Boost::getWeakCount()
    //

    /**
     * SEE: setWeakCount
     * @return automatically generated
     */
    public int getWeakCount() {
        return getWeakCount_0(nativeObj);
    }


    //
    // C++:  void cv::ml::Boost::setBoostType(int val)
    //

    /**
     *  getBoostType SEE: getBoostType
     * @param val automatically generated
     */
    public void setBoostType(int val) {
        setBoostType_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::Boost::setWeakCount(int val)
    //

    /**
     *  getWeakCount SEE: getWeakCount
     * @param val automatically generated
     */
    public void setWeakCount(int val) {
        setWeakCount_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::Boost::setWeightTrimRate(double val)
    //

    /**
     *  getWeightTrimRate SEE: getWeightTrimRate
     * @param val automatically generated
     */
    public void setWeightTrimRate(double val) {
        setWeightTrimRate_0(nativeObj, val);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_Boost cv::ml::Boost::create()
    private static native long create_0();

    // C++: static Ptr_Boost cv::ml::Boost::load(String filepath, String nodeName = String())
    private static native long load_0(String filepath, String nodeName);
    private static native long load_1(String filepath);

    // C++:  double cv::ml::Boost::getWeightTrimRate()
    private static native double getWeightTrimRate_0(long nativeObj);

    // C++:  int cv::ml::Boost::getBoostType()
    private static native int getBoostType_0(long nativeObj);

    // C++:  int cv::ml::Boost::getWeakCount()
    private static native int getWeakCount_0(long nativeObj);

    // C++:  void cv::ml::Boost::setBoostType(int val)
    private static native void setBoostType_0(long nativeObj, int val);

    // C++:  void cv::ml::Boost::setWeakCount(int val)
    private static native void setWeakCount_0(long nativeObj, int val);

    // C++:  void cv::ml::Boost::setWeightTrimRate(double val)
    private static native void setWeightTrimRate_0(long nativeObj, double val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
