//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.dnn;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.dnn.Model;
import org.opencv.dnn.Net;
import org.opencv.utils.Converters;

// C++: class KeypointsModel
/**
 * This class represents high-level API for keypoints models
 *
 * KeypointsModel allows to set params for preprocessing input image.
 * KeypointsModel creates net from file with trained weights and config,
 * sets preprocessing input, runs forward pass and returns the x and y coordinates of each detected keypoint
 */
public class KeypointsModel extends Model {

    protected KeypointsModel(long addr) { super(addr); }

    // internal usage only
    public static KeypointsModel __fromPtr__(long addr) { return new KeypointsModel(addr); }

    //
    // C++:   cv::dnn::KeypointsModel::KeypointsModel(Net network)
    //

    /**
     * Create model from deep learning network.
     * @param network Net object.
     */
    public KeypointsModel(Net network) {
        super(KeypointsModel_0(network.nativeObj));
    }


    //
    // C++:   cv::dnn::KeypointsModel::KeypointsModel(String model, String config = "")
    //

    /**
     * Create keypoints model from network represented in one of the supported formats.
     * An order of {@code model} and {@code config} arguments does not matter.
     * @param model Binary file contains trained weights.
     * @param config Text file contains network configuration.
     */
    public KeypointsModel(String model, String config) {
        super(KeypointsModel_1(model, config));
    }

    /**
     * Create keypoints model from network represented in one of the supported formats.
     * An order of {@code model} and {@code config} arguments does not matter.
     * @param model Binary file contains trained weights.
     */
    public KeypointsModel(String model) {
        super(KeypointsModel_2(model));
    }


    //
    // C++:  vector_Point2f cv::dnn::KeypointsModel::estimate(Mat frame, float thresh = 0.5)
    //

    /**
     * Given the {@code input} frame, create input blob, run net
     * @param thresh minimum confidence threshold to select a keypoint
     * @return a vector holding the x and y coordinates of each detected keypoint
     *
     * @param frame automatically generated
     */
    public MatOfPoint2f estimate(Mat frame, float thresh) {
        return MatOfPoint2f.fromNativeAddr(estimate_0(nativeObj, frame.nativeObj, thresh));
    }

    /**
     * Given the {@code input} frame, create input blob, run net
     * @return a vector holding the x and y coordinates of each detected keypoint
     *
     * @param frame automatically generated
     */
    public MatOfPoint2f estimate(Mat frame) {
        return MatOfPoint2f.fromNativeAddr(estimate_1(nativeObj, frame.nativeObj));
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::dnn::KeypointsModel::KeypointsModel(Net network)
    private static native long KeypointsModel_0(long network_nativeObj);

    // C++:   cv::dnn::KeypointsModel::KeypointsModel(String model, String config = "")
    private static native long KeypointsModel_1(String model, String config);
    private static native long KeypointsModel_2(String model);

    // C++:  vector_Point2f cv::dnn::KeypointsModel::estimate(Mat frame, float thresh = 0.5)
    private static native long estimate_0(long nativeObj, long frame_nativeObj, float thresh);
    private static native long estimate_1(long nativeObj, long frame_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
