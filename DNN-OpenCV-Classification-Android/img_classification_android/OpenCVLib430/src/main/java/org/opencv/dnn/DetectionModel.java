//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.dnn;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.dnn.Model;
import org.opencv.dnn.Net;
import org.opencv.utils.Converters;

// C++: class DetectionModel
/**
 * This class represents high-level API for object detection networks.
 *
 * DetectionModel allows to set params for preprocessing input image.
 * DetectionModel creates net from file with trained weights and config,
 * sets preprocessing input, runs forward pass and return result detections.
 * For DetectionModel SSD, Faster R-CNN, YOLO topologies are supported.
 */
public class DetectionModel extends Model {

    protected DetectionModel(long addr) { super(addr); }

    // internal usage only
    public static DetectionModel __fromPtr__(long addr) { return new DetectionModel(addr); }

    //
    // C++:   cv::dnn::DetectionModel::DetectionModel(Net network)
    //

    /**
     * Create model from deep learning network.
     * @param network Net object.
     */
    public DetectionModel(Net network) {
        super(DetectionModel_0(network.nativeObj));
    }


    //
    // C++:   cv::dnn::DetectionModel::DetectionModel(String model, String config = "")
    //

    /**
     * Create detection model from network represented in one of the supported formats.
     * An order of {@code model} and {@code config} arguments does not matter.
     * @param model Binary file contains trained weights.
     * @param config Text file contains network configuration.
     */
    public DetectionModel(String model, String config) {
        super(DetectionModel_1(model, config));
    }

    /**
     * Create detection model from network represented in one of the supported formats.
     * An order of {@code model} and {@code config} arguments does not matter.
     * @param model Binary file contains trained weights.
     */
    public DetectionModel(String model) {
        super(DetectionModel_2(model));
    }


    //
    // C++:  void cv::dnn::DetectionModel::detect(Mat frame, vector_int& classIds, vector_float& confidences, vector_Rect& boxes, float confThreshold = 0.5f, float nmsThreshold = 0.0f)
    //

    /**
     * Given the {@code input} frame, create input blob, run net and return result detections.
     * @param classIds Class indexes in result detection.
     * @param confidences A set of corresponding confidences.
     * @param boxes A set of bounding boxes.
     * @param confThreshold A threshold used to filter boxes by confidences.
     * @param nmsThreshold A threshold used in non maximum suppression.
     * @param frame automatically generated
     */
    public void detect(Mat frame, MatOfInt classIds, MatOfFloat confidences, MatOfRect boxes, float confThreshold, float nmsThreshold) {
        Mat classIds_mat = classIds;
        Mat confidences_mat = confidences;
        Mat boxes_mat = boxes;
        detect_0(nativeObj, frame.nativeObj, classIds_mat.nativeObj, confidences_mat.nativeObj, boxes_mat.nativeObj, confThreshold, nmsThreshold);
    }

    /**
     * Given the {@code input} frame, create input blob, run net and return result detections.
     * @param classIds Class indexes in result detection.
     * @param confidences A set of corresponding confidences.
     * @param boxes A set of bounding boxes.
     * @param confThreshold A threshold used to filter boxes by confidences.
     * @param frame automatically generated
     */
    public void detect(Mat frame, MatOfInt classIds, MatOfFloat confidences, MatOfRect boxes, float confThreshold) {
        Mat classIds_mat = classIds;
        Mat confidences_mat = confidences;
        Mat boxes_mat = boxes;
        detect_1(nativeObj, frame.nativeObj, classIds_mat.nativeObj, confidences_mat.nativeObj, boxes_mat.nativeObj, confThreshold);
    }

    /**
     * Given the {@code input} frame, create input blob, run net and return result detections.
     * @param classIds Class indexes in result detection.
     * @param confidences A set of corresponding confidences.
     * @param boxes A set of bounding boxes.
     * @param frame automatically generated
     */
    public void detect(Mat frame, MatOfInt classIds, MatOfFloat confidences, MatOfRect boxes) {
        Mat classIds_mat = classIds;
        Mat confidences_mat = confidences;
        Mat boxes_mat = boxes;
        detect_2(nativeObj, frame.nativeObj, classIds_mat.nativeObj, confidences_mat.nativeObj, boxes_mat.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::dnn::DetectionModel::DetectionModel(Net network)
    private static native long DetectionModel_0(long network_nativeObj);

    // C++:   cv::dnn::DetectionModel::DetectionModel(String model, String config = "")
    private static native long DetectionModel_1(String model, String config);
    private static native long DetectionModel_2(String model);

    // C++:  void cv::dnn::DetectionModel::detect(Mat frame, vector_int& classIds, vector_float& confidences, vector_Rect& boxes, float confThreshold = 0.5f, float nmsThreshold = 0.0f)
    private static native void detect_0(long nativeObj, long frame_nativeObj, long classIds_mat_nativeObj, long confidences_mat_nativeObj, long boxes_mat_nativeObj, float confThreshold, float nmsThreshold);
    private static native void detect_1(long nativeObj, long frame_nativeObj, long classIds_mat_nativeObj, long confidences_mat_nativeObj, long boxes_mat_nativeObj, float confThreshold);
    private static native void detect_2(long nativeObj, long frame_nativeObj, long classIds_mat_nativeObj, long confidences_mat_nativeObj, long boxes_mat_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
