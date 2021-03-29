//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.dnn;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

// C++: class Layer
/**
 * This interface class allows to build new Layers - are building blocks of networks.
 *
 * Each class, derived from Layer, must implement allocate() methods to declare own outputs and forward() to compute outputs.
 * Also before using the new layer into networks you must register your layer by using one of REF: dnnLayerFactory "LayerFactory" macros.
 */
public class Layer extends Algorithm {

    protected Layer(long addr) { super(addr); }

    // internal usage only
    public static Layer __fromPtr__(long addr) { return new Layer(addr); }

    //
    // C++:  int cv::dnn::Layer::outputNameToIndex(String outputName)
    //

    /**
     * Returns index of output blob in output array.
     * SEE: inputNameToIndex()
     * @param outputName automatically generated
     * @return automatically generated
     */
    public int outputNameToIndex(String outputName) {
        return outputNameToIndex_0(nativeObj, outputName);
    }


    //
    // C++:  void cv::dnn::Layer::finalize(vector_Mat inputs, vector_Mat& outputs)
    //

    /**
     * Computes and sets internal parameters according to inputs, outputs and blobs.
     * @param outputs vector of already allocated output blobs
     *
     * If this method is called after network has allocated all memory for input and output blobs
     * and before inferencing.
     * @param inputs automatically generated
     */
    public void finalize(List<Mat> inputs, List<Mat> outputs) {
        Mat inputs_mat = Converters.vector_Mat_to_Mat(inputs);
        Mat outputs_mat = new Mat();
        finalize_0(nativeObj, inputs_mat.nativeObj, outputs_mat.nativeObj);
        Converters.Mat_to_vector_Mat(outputs_mat, outputs);
        outputs_mat.release();
    }


    //
    // C++:  void cv::dnn::Layer::run(vector_Mat inputs, vector_Mat& outputs, vector_Mat& internals)
    //

    /**
     * Allocates layer and computes output.
     * @deprecated This method will be removed in the future release.
     * @param inputs automatically generated
     * @param outputs automatically generated
     * @param internals automatically generated
     */
    @Deprecated
    public void run(List<Mat> inputs, List<Mat> outputs, List<Mat> internals) {
        Mat inputs_mat = Converters.vector_Mat_to_Mat(inputs);
        Mat outputs_mat = new Mat();
        Mat internals_mat = Converters.vector_Mat_to_Mat(internals);
        run_0(nativeObj, inputs_mat.nativeObj, outputs_mat.nativeObj, internals_mat.nativeObj);
        Converters.Mat_to_vector_Mat(outputs_mat, outputs);
        outputs_mat.release();
        Converters.Mat_to_vector_Mat(internals_mat, internals);
        internals_mat.release();
    }


    //
    // C++: vector_Mat Layer::blobs
    //

    public List<Mat> get_blobs() {
        List<Mat> retVal = new ArrayList<Mat>();
        Mat retValMat = new Mat(get_blobs_0(nativeObj));
        Converters.Mat_to_vector_Mat(retValMat, retVal);
        return retVal;
    }


    //
    // C++: void Layer::blobs
    //

    public void set_blobs(List<Mat> blobs) {
        Mat blobs_mat = Converters.vector_Mat_to_Mat(blobs);
        set_blobs_0(nativeObj, blobs_mat.nativeObj);
    }


    //
    // C++: String Layer::name
    //

    public String get_name() {
        return get_name_0(nativeObj);
    }


    //
    // C++: String Layer::type
    //

    public String get_type() {
        return get_type_0(nativeObj);
    }


    //
    // C++: int Layer::preferableTarget
    //

    public int get_preferableTarget() {
        return get_preferableTarget_0(nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  int cv::dnn::Layer::outputNameToIndex(String outputName)
    private static native int outputNameToIndex_0(long nativeObj, String outputName);

    // C++:  void cv::dnn::Layer::finalize(vector_Mat inputs, vector_Mat& outputs)
    private static native void finalize_0(long nativeObj, long inputs_mat_nativeObj, long outputs_mat_nativeObj);

    // C++:  void cv::dnn::Layer::run(vector_Mat inputs, vector_Mat& outputs, vector_Mat& internals)
    private static native void run_0(long nativeObj, long inputs_mat_nativeObj, long outputs_mat_nativeObj, long internals_mat_nativeObj);

    // C++: vector_Mat Layer::blobs
    private static native long get_blobs_0(long nativeObj);

    // C++: void Layer::blobs
    private static native void set_blobs_0(long nativeObj, long blobs_mat_nativeObj);

    // C++: String Layer::name
    private static native String get_name_0(long nativeObj);

    // C++: String Layer::type
    private static native String get_type_0(long nativeObj);

    // C++: int Layer::preferableTarget
    private static native int get_preferableTarget_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
