//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.ml.NormalBayesClassifier;
import org.opencv.ml.StatModel;

// C++: class NormalBayesClassifier
/**
 * Bayes classifier for normally distributed data.
 *
 * SEE: REF: ml_intro_bayes
 */
public class NormalBayesClassifier extends StatModel {

    protected NormalBayesClassifier(long addr) { super(addr); }

    // internal usage only
    public static NormalBayesClassifier __fromPtr__(long addr) { return new NormalBayesClassifier(addr); }

    //
    // C++: static Ptr_NormalBayesClassifier cv::ml::NormalBayesClassifier::create()
    //

    /**
     * Creates empty model
     * Use StatModel::train to train the model after creation.
     * @return automatically generated
     */
    public static NormalBayesClassifier create() {
        return NormalBayesClassifier.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_NormalBayesClassifier cv::ml::NormalBayesClassifier::load(String filepath, String nodeName = String())
    //

    /**
     * Loads and creates a serialized NormalBayesClassifier from a file
     *
     * Use NormalBayesClassifier::save to serialize and store an NormalBayesClassifier to disk.
     * Load the NormalBayesClassifier from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized NormalBayesClassifier
     * @param nodeName name of node containing the classifier
     * @return automatically generated
     */
    public static NormalBayesClassifier load(String filepath, String nodeName) {
        return NormalBayesClassifier.__fromPtr__(load_0(filepath, nodeName));
    }

    /**
     * Loads and creates a serialized NormalBayesClassifier from a file
     *
     * Use NormalBayesClassifier::save to serialize and store an NormalBayesClassifier to disk.
     * Load the NormalBayesClassifier from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized NormalBayesClassifier
     * @return automatically generated
     */
    public static NormalBayesClassifier load(String filepath) {
        return NormalBayesClassifier.__fromPtr__(load_1(filepath));
    }


    //
    // C++:  float cv::ml::NormalBayesClassifier::predictProb(Mat inputs, Mat& outputs, Mat& outputProbs, int flags = 0)
    //

    /**
     * Predicts the response for sample(s).
     *
     *     The method estimates the most probable classes for input vectors. Input vectors (one or more)
     *     are stored as rows of the matrix inputs. In case of multiple input vectors, there should be one
     *     output vector outputs. The predicted class for a single input vector is returned by the method.
     *     The vector outputProbs contains the output probabilities corresponding to each element of
     *     result.
     * @param inputs automatically generated
     * @param outputs automatically generated
     * @param outputProbs automatically generated
     * @param flags automatically generated
     * @return automatically generated
     */
    public float predictProb(Mat inputs, Mat outputs, Mat outputProbs, int flags) {
        return predictProb_0(nativeObj, inputs.nativeObj, outputs.nativeObj, outputProbs.nativeObj, flags);
    }

    /**
     * Predicts the response for sample(s).
     *
     *     The method estimates the most probable classes for input vectors. Input vectors (one or more)
     *     are stored as rows of the matrix inputs. In case of multiple input vectors, there should be one
     *     output vector outputs. The predicted class for a single input vector is returned by the method.
     *     The vector outputProbs contains the output probabilities corresponding to each element of
     *     result.
     * @param inputs automatically generated
     * @param outputs automatically generated
     * @param outputProbs automatically generated
     * @return automatically generated
     */
    public float predictProb(Mat inputs, Mat outputs, Mat outputProbs) {
        return predictProb_1(nativeObj, inputs.nativeObj, outputs.nativeObj, outputProbs.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_NormalBayesClassifier cv::ml::NormalBayesClassifier::create()
    private static native long create_0();

    // C++: static Ptr_NormalBayesClassifier cv::ml::NormalBayesClassifier::load(String filepath, String nodeName = String())
    private static native long load_0(String filepath, String nodeName);
    private static native long load_1(String filepath);

    // C++:  float cv::ml::NormalBayesClassifier::predictProb(Mat inputs, Mat& outputs, Mat& outputProbs, int flags = 0)
    private static native float predictProb_0(long nativeObj, long inputs_nativeObj, long outputs_nativeObj, long outputProbs_nativeObj, int flags);
    private static native float predictProb_1(long nativeObj, long inputs_nativeObj, long outputs_nativeObj, long outputProbs_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
