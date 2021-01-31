//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.dnn;

import org.opencv.core.Mat;
import org.opencv.dnn.Model;
import org.opencv.dnn.Net;

// C++: class SegmentationModel
/**
 * This class represents high-level API for segmentation  models
 *
 * SegmentationModel allows to set params for preprocessing input image.
 * SegmentationModel creates net from file with trained weights and config,
 * sets preprocessing input, runs forward pass and returns the class prediction for each pixel.
 */
public class SegmentationModel extends Model {

    protected SegmentationModel(long addr) { super(addr); }

    // internal usage only
    public static SegmentationModel __fromPtr__(long addr) { return new SegmentationModel(addr); }

    //
    // C++:   cv::dnn::SegmentationModel::SegmentationModel(Net network)
    //

    /**
     * Create model from deep learning network.
     * @param network Net object.
     */
    public SegmentationModel(Net network) {
        super(SegmentationModel_0(network.nativeObj));
    }


    //
    // C++:   cv::dnn::SegmentationModel::SegmentationModel(String model, String config = "")
    //

    /**
     * Create segmentation model from network represented in one of the supported formats.
     * An order of {@code model} and {@code config} arguments does not matter.
     * @param model Binary file contains trained weights.
     * @param config Text file contains network configuration.
     */
    public SegmentationModel(String model, String config) {
        super(SegmentationModel_1(model, config));
    }

    /**
     * Create segmentation model from network represented in one of the supported formats.
     * An order of {@code model} and {@code config} arguments does not matter.
     * @param model Binary file contains trained weights.
     */
    public SegmentationModel(String model) {
        super(SegmentationModel_2(model));
    }


    //
    // C++:  void cv::dnn::SegmentationModel::segment(Mat frame, Mat& mask)
    //

    /**
     * Given the {@code input} frame, create input blob, run net
     * @param mask Allocated class prediction for each pixel
     * @param frame automatically generated
     */
    public void segment(Mat frame, Mat mask) {
        segment_0(nativeObj, frame.nativeObj, mask.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::dnn::SegmentationModel::SegmentationModel(Net network)
    private static native long SegmentationModel_0(long network_nativeObj);

    // C++:   cv::dnn::SegmentationModel::SegmentationModel(String model, String config = "")
    private static native long SegmentationModel_1(String model, String config);
    private static native long SegmentationModel_2(String model);

    // C++:  void cv::dnn::SegmentationModel::segment(Mat frame, Mat& mask)
    private static native void segment_0(long nativeObj, long frame_nativeObj, long mask_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
