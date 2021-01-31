//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.dnn;

import org.opencv.core.Mat;
import org.opencv.dnn.Model;
import org.opencv.dnn.Net;

// C++: class ClassificationModel
/**
 * This class represents high-level API for classification models.
 *
 * ClassificationModel allows to set params for preprocessing input image.
 * ClassificationModel creates net from file with trained weights and config,
 * sets preprocessing input, runs forward pass and return top-1 prediction.
 */
public class ClassificationModel extends Model {

    protected ClassificationModel(long addr) { super(addr); }

    // internal usage only
    public static ClassificationModel __fromPtr__(long addr) { return new ClassificationModel(addr); }

    //
    // C++:   cv::dnn::ClassificationModel::ClassificationModel(Net network)
    //

    /**
     * Create model from deep learning network.
     * @param network Net object.
     */
    public ClassificationModel(Net network) {
        super(ClassificationModel_0(network.nativeObj));
    }


    //
    // C++:   cv::dnn::ClassificationModel::ClassificationModel(String model, String config = "")
    //

    /**
     * Create classification model from network represented in one of the supported formats.
     * An order of {@code model} and {@code config} arguments does not matter.
     * @param model Binary file contains trained weights.
     * @param config Text file contains network configuration.
     */
    public ClassificationModel(String model, String config) {
        super(ClassificationModel_1(model, config));
    }

    /**
     * Create classification model from network represented in one of the supported formats.
     * An order of {@code model} and {@code config} arguments does not matter.
     * @param model Binary file contains trained weights.
     */
    public ClassificationModel(String model) {
        super(ClassificationModel_2(model));
    }


    //
    // C++:  void cv::dnn::ClassificationModel::classify(Mat frame, int& classId, float& conf)
    //

    public void classify(Mat frame, int[] classId, float[] conf) {
        double[] classId_out = new double[1];
        double[] conf_out = new double[1];
        classify_0(nativeObj, frame.nativeObj, classId_out, conf_out);
        if(classId!=null) classId[0] = (int)classId_out[0];
        if(conf!=null) conf[0] = (float)conf_out[0];
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::dnn::ClassificationModel::ClassificationModel(Net network)
    private static native long ClassificationModel_0(long network_nativeObj);

    // C++:   cv::dnn::ClassificationModel::ClassificationModel(String model, String config = "")
    private static native long ClassificationModel_1(String model, String config);
    private static native long ClassificationModel_2(String model);

    // C++:  void cv::dnn::ClassificationModel::classify(Mat frame, int& classId, float& conf)
    private static native void classify_0(long nativeObj, long frame_nativeObj, double[] classId_out, double[] conf_out);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
