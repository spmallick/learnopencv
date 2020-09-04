//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.dnn;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.dnn.DictValue;
import org.opencv.dnn.Layer;
import org.opencv.dnn.Net;
import org.opencv.utils.Converters;

// C++: class Net
/**
 * This class allows to create and manipulate comprehensive artificial neural networks.
 *
 * Neural network is presented as directed acyclic graph (DAG), where vertices are Layer instances,
 * and edges specify relationships between layers inputs and outputs.
 *
 * Each network layer has unique integer id and unique string name inside its network.
 * LayerId can store either layer name or layer id.
 *
 * This class supports reference counting of its instances, i. e. copies point to the same instance.
 */
public class Net {

    protected final long nativeObj;
    protected Net(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static Net __fromPtr__(long addr) { return new Net(addr); }

    //
    // C++:   cv::dnn::Net::Net()
    //

    public Net() {
        nativeObj = Net_0();
    }


    //
    // C++:  AsyncArray cv::dnn::Net::forwardAsync(String outputName = String())
    //

    // Return type 'AsyncArray' is not supported, skipping the function


    //
    // C++:  Mat cv::dnn::Net::forward(String outputName = String())
    //

    /**
     * Runs forward pass to compute output of layer with name {@code outputName}.
     * @param outputName name for layer which output is needed to get
     * @return blob for first output of specified layer.
     * By default runs forward pass for the whole network.
     */
    public Mat forward(String outputName) {
        return new Mat(forward_0(nativeObj, outputName));
    }

    /**
     * Runs forward pass to compute output of layer with name {@code outputName}.
     * @return blob for first output of specified layer.
     * By default runs forward pass for the whole network.
     */
    public Mat forward() {
        return new Mat(forward_1(nativeObj));
    }


    //
    // C++:  Mat cv::dnn::Net::getParam(LayerId layer, int numParam = 0)
    //

    /**
     * Returns parameter blob of the layer.
     * @param layer name or id of the layer.
     * @param numParam index of the layer parameter in the Layer::blobs array.
     * SEE: Layer::blobs
     * @return automatically generated
     */
    public Mat getParam(DictValue layer, int numParam) {
        return new Mat(getParam_0(nativeObj, layer.getNativeObjAddr(), numParam));
    }

    /**
     * Returns parameter blob of the layer.
     * @param layer name or id of the layer.
     * SEE: Layer::blobs
     * @return automatically generated
     */
    public Mat getParam(DictValue layer) {
        return new Mat(getParam_1(nativeObj, layer.getNativeObjAddr()));
    }


    //
    // C++: static Net cv::dnn::Net::readFromModelOptimizer(String xml, String bin)
    //

    /**
     * Create a network from Intel's Model Optimizer intermediate representation (IR).
     * @param xml XML configuration file with network's topology.
     * @param bin Binary file with trained weights.
     * Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
     * backend.
     * @return automatically generated
     */
    public static Net readFromModelOptimizer(String xml, String bin) {
        return new Net(readFromModelOptimizer_0(xml, bin));
    }


    //
    // C++: static Net cv::dnn::Net::readFromModelOptimizer(vector_uchar bufferModelConfig, vector_uchar bufferWeights)
    //

    /**
     * Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).
     * @param bufferModelConfig buffer with model's configuration.
     * @param bufferWeights buffer with model's trained weights.
     * @return Net object.
     */
    public static Net readFromModelOptimizer(MatOfByte bufferModelConfig, MatOfByte bufferWeights) {
        Mat bufferModelConfig_mat = bufferModelConfig;
        Mat bufferWeights_mat = bufferWeights;
        return new Net(readFromModelOptimizer_1(bufferModelConfig_mat.nativeObj, bufferWeights_mat.nativeObj));
    }


    //
    // C++:  Ptr_Layer cv::dnn::Net::getLayer(LayerId layerId)
    //

    /**
     * Returns pointer to layer with specified id or name which the network use.
     * @param layerId automatically generated
     * @return automatically generated
     */
    public Layer getLayer(DictValue layerId) {
        return Layer.__fromPtr__(getLayer_0(nativeObj, layerId.getNativeObjAddr()));
    }


    //
    // C++:  String cv::dnn::Net::dump()
    //

    /**
     * Dump net to String
     * @return String with structure, hyperparameters, backend, target and fusion
     * Call method after setInput(). To see correct backend, target and fusion run after forward().
     */
    public String dump() {
        return dump_0(nativeObj);
    }


    //
    // C++:  bool cv::dnn::Net::empty()
    //

    /**
     * Returns true if there are no layers in the network.
     * @return automatically generated
     */
    public boolean empty() {
        return empty_0(nativeObj);
    }


    //
    // C++:  int cv::dnn::Net::getLayerId(String layer)
    //

    /**
     * Converts string name of the layer to the integer identifier.
     * @return id of the layer, or -1 if the layer wasn't found.
     * @param layer automatically generated
     */
    public int getLayerId(String layer) {
        return getLayerId_0(nativeObj, layer);
    }


    //
    // C++:  int cv::dnn::Net::getLayersCount(String layerType)
    //

    /**
     * Returns count of layers of specified type.
     * @param layerType type.
     * @return count of layers
     */
    public int getLayersCount(String layerType) {
        return getLayersCount_0(nativeObj, layerType);
    }


    //
    // C++:  int64 cv::dnn::Net::getFLOPS(MatShape netInputShape)
    //

    public long getFLOPS(MatOfInt netInputShape) {
        Mat netInputShape_mat = netInputShape;
        return getFLOPS_0(nativeObj, netInputShape_mat.nativeObj);
    }


    //
    // C++:  int64 cv::dnn::Net::getFLOPS(int layerId, MatShape netInputShape)
    //

    public long getFLOPS(int layerId, MatOfInt netInputShape) {
        Mat netInputShape_mat = netInputShape;
        return getFLOPS_1(nativeObj, layerId, netInputShape_mat.nativeObj);
    }


    //
    // C++:  int64 cv::dnn::Net::getFLOPS(int layerId, vector_MatShape netInputShapes)
    //

    public long getFLOPS(int layerId, List<MatOfInt> netInputShapes) {
        return getFLOPS_2(nativeObj, layerId, netInputShapes);
    }


    //
    // C++:  int64 cv::dnn::Net::getFLOPS(vector_MatShape netInputShapes)
    //

    /**
     * Computes FLOP for whole loaded model with specified input shapes.
     * @param netInputShapes vector of shapes for all net inputs.
     * @return computed FLOP.
     */
    public long getFLOPS(List<MatOfInt> netInputShapes) {
        return getFLOPS_3(nativeObj, netInputShapes);
    }


    //
    // C++:  int64 cv::dnn::Net::getPerfProfile(vector_double& timings)
    //

    /**
     * Returns overall time for inference and timings (in ticks) for layers.
     * Indexes in returned vector correspond to layers ids. Some layers can be fused with others,
     * in this case zero ticks count will be return for that skipped layers.
     * @param timings vector for tick timings for all layers.
     * @return overall ticks for model inference.
     */
    public long getPerfProfile(MatOfDouble timings) {
        Mat timings_mat = timings;
        return getPerfProfile_0(nativeObj, timings_mat.nativeObj);
    }


    //
    // C++:  vector_String cv::dnn::Net::getLayerNames()
    //

    public List<String> getLayerNames() {
        return getLayerNames_0(nativeObj);
    }


    //
    // C++:  vector_String cv::dnn::Net::getUnconnectedOutLayersNames()
    //

    /**
     * Returns names of layers with unconnected outputs.
     * @return automatically generated
     */
    public List<String> getUnconnectedOutLayersNames() {
        return getUnconnectedOutLayersNames_0(nativeObj);
    }


    //
    // C++:  vector_int cv::dnn::Net::getUnconnectedOutLayers()
    //

    /**
     * Returns indexes of layers with unconnected outputs.
     * @return automatically generated
     */
    public MatOfInt getUnconnectedOutLayers() {
        return MatOfInt.fromNativeAddr(getUnconnectedOutLayers_0(nativeObj));
    }


    //
    // C++:  void cv::dnn::Net::connect(String outPin, String inpPin)
    //

    /**
     * Connects output of the first layer to input of the second layer.
     * @param outPin descriptor of the first layer output.
     * @param inpPin descriptor of the second layer input.
     *
     * Descriptors have the following template &lt;DFN&gt;&amp;lt;layer_name&amp;gt;[.input_number]&lt;/DFN&gt;:
     * - the first part of the template &lt;DFN&gt;layer_name&lt;/DFN&gt; is string name of the added layer.
     * If this part is empty then the network input pseudo layer will be used;
     * - the second optional part of the template &lt;DFN&gt;input_number&lt;/DFN&gt;
     * is either number of the layer input, either label one.
     * If this part is omitted then the first layer input will be used.
     *
     * SEE: setNetInputs(), Layer::inputNameToIndex(), Layer::outputNameToIndex()
     */
    public void connect(String outPin, String inpPin) {
        connect_0(nativeObj, outPin, inpPin);
    }


    //
    // C++:  void cv::dnn::Net::dumpToFile(String path)
    //

    /**
     * Dump net structure, hyperparameters, backend, target and fusion to dot file
     * @param path   path to output file with .dot extension
     * SEE: dump()
     */
    public void dumpToFile(String path) {
        dumpToFile_0(nativeObj, path);
    }


    //
    // C++:  void cv::dnn::Net::enableFusion(bool fusion)
    //

    /**
     * Enables or disables layer fusion in the network.
     * @param fusion true to enable the fusion, false to disable. The fusion is enabled by default.
     */
    public void enableFusion(boolean fusion) {
        enableFusion_0(nativeObj, fusion);
    }


    //
    // C++:  void cv::dnn::Net::forward(vector_Mat& outputBlobs, String outputName = String())
    //

    /**
     * Runs forward pass to compute output of layer with name {@code outputName}.
     * @param outputBlobs contains all output blobs for specified layer.
     * @param outputName name for layer which output is needed to get
     * If {@code outputName} is empty, runs forward pass for the whole network.
     */
    public void forward(List<Mat> outputBlobs, String outputName) {
        Mat outputBlobs_mat = new Mat();
        forward_2(nativeObj, outputBlobs_mat.nativeObj, outputName);
        Converters.Mat_to_vector_Mat(outputBlobs_mat, outputBlobs);
        outputBlobs_mat.release();
    }

    /**
     * Runs forward pass to compute output of layer with name {@code outputName}.
     * @param outputBlobs contains all output blobs for specified layer.
     * If {@code outputName} is empty, runs forward pass for the whole network.
     */
    public void forward(List<Mat> outputBlobs) {
        Mat outputBlobs_mat = new Mat();
        forward_3(nativeObj, outputBlobs_mat.nativeObj);
        Converters.Mat_to_vector_Mat(outputBlobs_mat, outputBlobs);
        outputBlobs_mat.release();
    }


    //
    // C++:  void cv::dnn::Net::forward(vector_Mat& outputBlobs, vector_String outBlobNames)
    //

    /**
     * Runs forward pass to compute outputs of layers listed in {@code outBlobNames}.
     * @param outputBlobs contains blobs for first outputs of specified layers.
     * @param outBlobNames names for layers which outputs are needed to get
     */
    public void forward(List<Mat> outputBlobs, List<String> outBlobNames) {
        Mat outputBlobs_mat = new Mat();
        forward_4(nativeObj, outputBlobs_mat.nativeObj, outBlobNames);
        Converters.Mat_to_vector_Mat(outputBlobs_mat, outputBlobs);
        outputBlobs_mat.release();
    }


    //
    // C++:  void cv::dnn::Net::forward(vector_vector_Mat& outputBlobs, vector_String outBlobNames)
    //

    // Unknown type 'vector_vector_Mat' (O), skipping the function


    //
    // C++:  void cv::dnn::Net::getLayerTypes(vector_String& layersTypes)
    //

    /**
     * Returns list of types for layer used in model.
     * @param layersTypes output parameter for returning types.
     */
    public void getLayerTypes(List<String> layersTypes) {
        getLayerTypes_0(nativeObj, layersTypes);
    }


    //
    // C++:  void cv::dnn::Net::getLayersShapes(MatShape netInputShape, vector_int& layersIds, vector_vector_MatShape& inLayersShapes, vector_vector_MatShape& outLayersShapes)
    //

    // Unknown type 'vector_vector_MatShape' (O), skipping the function


    //
    // C++:  void cv::dnn::Net::getLayersShapes(vector_MatShape netInputShapes, vector_int& layersIds, vector_vector_MatShape& inLayersShapes, vector_vector_MatShape& outLayersShapes)
    //

    // Unknown type 'vector_vector_MatShape' (O), skipping the function


    //
    // C++:  void cv::dnn::Net::getMemoryConsumption(MatShape netInputShape, size_t& weights, size_t& blobs)
    //

    public void getMemoryConsumption(MatOfInt netInputShape, long[] weights, long[] blobs) {
        Mat netInputShape_mat = netInputShape;
        double[] weights_out = new double[1];
        double[] blobs_out = new double[1];
        getMemoryConsumption_0(nativeObj, netInputShape_mat.nativeObj, weights_out, blobs_out);
        if(weights!=null) weights[0] = (long)weights_out[0];
        if(blobs!=null) blobs[0] = (long)blobs_out[0];
    }


    //
    // C++:  void cv::dnn::Net::getMemoryConsumption(int layerId, MatShape netInputShape, size_t& weights, size_t& blobs)
    //

    public void getMemoryConsumption(int layerId, MatOfInt netInputShape, long[] weights, long[] blobs) {
        Mat netInputShape_mat = netInputShape;
        double[] weights_out = new double[1];
        double[] blobs_out = new double[1];
        getMemoryConsumption_1(nativeObj, layerId, netInputShape_mat.nativeObj, weights_out, blobs_out);
        if(weights!=null) weights[0] = (long)weights_out[0];
        if(blobs!=null) blobs[0] = (long)blobs_out[0];
    }


    //
    // C++:  void cv::dnn::Net::getMemoryConsumption(int layerId, vector_MatShape netInputShapes, size_t& weights, size_t& blobs)
    //

    public void getMemoryConsumption(int layerId, List<MatOfInt> netInputShapes, long[] weights, long[] blobs) {
        double[] weights_out = new double[1];
        double[] blobs_out = new double[1];
        getMemoryConsumption_2(nativeObj, layerId, netInputShapes, weights_out, blobs_out);
        if(weights!=null) weights[0] = (long)weights_out[0];
        if(blobs!=null) blobs[0] = (long)blobs_out[0];
    }


    //
    // C++:  void cv::dnn::Net::setHalideScheduler(String scheduler)
    //

    /**
     * Compile Halide layers.
     * @param scheduler Path to YAML file with scheduling directives.
     * SEE: setPreferableBackend
     *
     * Schedule layers that support Halide backend. Then compile them for
     * specific target. For layers that not represented in scheduling file
     * or if no manual scheduling used at all, automatic scheduling will be applied.
     */
    public void setHalideScheduler(String scheduler) {
        setHalideScheduler_0(nativeObj, scheduler);
    }


    //
    // C++:  void cv::dnn::Net::setInput(Mat blob, String name = "", double scalefactor = 1.0, Scalar mean = Scalar())
    //

    /**
     * Sets the new input value for the network
     * @param blob        A new blob. Should have CV_32F or CV_8U depth.
     * @param name        A name of input layer.
     * @param scalefactor An optional normalization scale.
     * @param mean        An optional mean subtraction values.
     * SEE: connect(String, String) to know format of the descriptor.
     *
     * If scale or mean values are specified, a final input blob is computed
     * as:
     * \(input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)\)
     */
    public void setInput(Mat blob, String name, double scalefactor, Scalar mean) {
        setInput_0(nativeObj, blob.nativeObj, name, scalefactor, mean.val[0], mean.val[1], mean.val[2], mean.val[3]);
    }

    /**
     * Sets the new input value for the network
     * @param blob        A new blob. Should have CV_32F or CV_8U depth.
     * @param name        A name of input layer.
     * @param scalefactor An optional normalization scale.
     * SEE: connect(String, String) to know format of the descriptor.
     *
     * If scale or mean values are specified, a final input blob is computed
     * as:
     * \(input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)\)
     */
    public void setInput(Mat blob, String name, double scalefactor) {
        setInput_1(nativeObj, blob.nativeObj, name, scalefactor);
    }

    /**
     * Sets the new input value for the network
     * @param blob        A new blob. Should have CV_32F or CV_8U depth.
     * @param name        A name of input layer.
     * SEE: connect(String, String) to know format of the descriptor.
     *
     * If scale or mean values are specified, a final input blob is computed
     * as:
     * \(input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)\)
     */
    public void setInput(Mat blob, String name) {
        setInput_2(nativeObj, blob.nativeObj, name);
    }

    /**
     * Sets the new input value for the network
     * @param blob        A new blob. Should have CV_32F or CV_8U depth.
     * SEE: connect(String, String) to know format of the descriptor.
     *
     * If scale or mean values are specified, a final input blob is computed
     * as:
     * \(input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)\)
     */
    public void setInput(Mat blob) {
        setInput_3(nativeObj, blob.nativeObj);
    }


    //
    // C++:  void cv::dnn::Net::setInputShape(String inputName, MatShape shape)
    //

    /**
     * Specify shape of network input.
     * @param inputName automatically generated
     * @param shape automatically generated
     */
    public void setInputShape(String inputName, MatOfInt shape) {
        Mat shape_mat = shape;
        setInputShape_0(nativeObj, inputName, shape_mat.nativeObj);
    }


    //
    // C++:  void cv::dnn::Net::setInputsNames(vector_String inputBlobNames)
    //

    /**
     * Sets outputs names of the network input pseudo layer.
     *
     * Each net always has special own the network input pseudo layer with id=0.
     * This layer stores the user blobs only and don't make any computations.
     * In fact, this layer provides the only way to pass user data into the network.
     * As any other layer, this layer can label its outputs and this function provides an easy way to do this.
     * @param inputBlobNames automatically generated
     */
    public void setInputsNames(List<String> inputBlobNames) {
        setInputsNames_0(nativeObj, inputBlobNames);
    }


    //
    // C++:  void cv::dnn::Net::setParam(LayerId layer, int numParam, Mat blob)
    //

    /**
     * Sets the new value for the learned param of the layer.
     * @param layer name or id of the layer.
     * @param numParam index of the layer parameter in the Layer::blobs array.
     * @param blob the new value.
     * SEE: Layer::blobs
     * <b>Note:</b> If shape of the new blob differs from the previous shape,
     * then the following forward pass may fail.
     */
    public void setParam(DictValue layer, int numParam, Mat blob) {
        setParam_0(nativeObj, layer.getNativeObjAddr(), numParam, blob.nativeObj);
    }


    //
    // C++:  void cv::dnn::Net::setPreferableBackend(int backendId)
    //

    /**
     * Ask network to use specific computation backend where it supported.
     * @param backendId backend identifier.
     * SEE: Backend
     *
     * If OpenCV is compiled with Intel's Inference Engine library, DNN_BACKEND_DEFAULT
     * means DNN_BACKEND_INFERENCE_ENGINE. Otherwise it equals to DNN_BACKEND_OPENCV.
     */
    public void setPreferableBackend(int backendId) {
        setPreferableBackend_0(nativeObj, backendId);
    }


    //
    // C++:  void cv::dnn::Net::setPreferableTarget(int targetId)
    //

    /**
     * Ask network to make computations on specific target device.
     * @param targetId target identifier.
     * SEE: Target
     *
     * List of supported combinations backend / target:
     * |                        | DNN_BACKEND_OPENCV | DNN_BACKEND_INFERENCE_ENGINE | DNN_BACKEND_HALIDE |  DNN_BACKEND_CUDA |
     * |------------------------|--------------------|------------------------------|--------------------|-------------------|
     * | DNN_TARGET_CPU         |                  + |                            + |                  + |                   |
     * | DNN_TARGET_OPENCL      |                  + |                            + |                  + |                   |
     * | DNN_TARGET_OPENCL_FP16 |                  + |                            + |                    |                   |
     * | DNN_TARGET_MYRIAD      |                    |                            + |                    |                   |
     * | DNN_TARGET_FPGA        |                    |                            + |                    |                   |
     * | DNN_TARGET_CUDA        |                    |                              |                    |                 + |
     * | DNN_TARGET_CUDA_FP16   |                    |                              |                    |                 + |
     */
    public void setPreferableTarget(int targetId) {
        setPreferableTarget_0(nativeObj, targetId);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::dnn::Net::Net()
    private static native long Net_0();

    // C++:  Mat cv::dnn::Net::forward(String outputName = String())
    private static native long forward_0(long nativeObj, String outputName);
    private static native long forward_1(long nativeObj);

    // C++:  Mat cv::dnn::Net::getParam(LayerId layer, int numParam = 0)
    private static native long getParam_0(long nativeObj, long layer_nativeObj, int numParam);
    private static native long getParam_1(long nativeObj, long layer_nativeObj);

    // C++: static Net cv::dnn::Net::readFromModelOptimizer(String xml, String bin)
    private static native long readFromModelOptimizer_0(String xml, String bin);

    // C++: static Net cv::dnn::Net::readFromModelOptimizer(vector_uchar bufferModelConfig, vector_uchar bufferWeights)
    private static native long readFromModelOptimizer_1(long bufferModelConfig_mat_nativeObj, long bufferWeights_mat_nativeObj);

    // C++:  Ptr_Layer cv::dnn::Net::getLayer(LayerId layerId)
    private static native long getLayer_0(long nativeObj, long layerId_nativeObj);

    // C++:  String cv::dnn::Net::dump()
    private static native String dump_0(long nativeObj);

    // C++:  bool cv::dnn::Net::empty()
    private static native boolean empty_0(long nativeObj);

    // C++:  int cv::dnn::Net::getLayerId(String layer)
    private static native int getLayerId_0(long nativeObj, String layer);

    // C++:  int cv::dnn::Net::getLayersCount(String layerType)
    private static native int getLayersCount_0(long nativeObj, String layerType);

    // C++:  int64 cv::dnn::Net::getFLOPS(MatShape netInputShape)
    private static native long getFLOPS_0(long nativeObj, long netInputShape_mat_nativeObj);

    // C++:  int64 cv::dnn::Net::getFLOPS(int layerId, MatShape netInputShape)
    private static native long getFLOPS_1(long nativeObj, int layerId, long netInputShape_mat_nativeObj);

    // C++:  int64 cv::dnn::Net::getFLOPS(int layerId, vector_MatShape netInputShapes)
    private static native long getFLOPS_2(long nativeObj, int layerId, List<MatOfInt> netInputShapes);

    // C++:  int64 cv::dnn::Net::getFLOPS(vector_MatShape netInputShapes)
    private static native long getFLOPS_3(long nativeObj, List<MatOfInt> netInputShapes);

    // C++:  int64 cv::dnn::Net::getPerfProfile(vector_double& timings)
    private static native long getPerfProfile_0(long nativeObj, long timings_mat_nativeObj);

    // C++:  vector_String cv::dnn::Net::getLayerNames()
    private static native List<String> getLayerNames_0(long nativeObj);

    // C++:  vector_String cv::dnn::Net::getUnconnectedOutLayersNames()
    private static native List<String> getUnconnectedOutLayersNames_0(long nativeObj);

    // C++:  vector_int cv::dnn::Net::getUnconnectedOutLayers()
    private static native long getUnconnectedOutLayers_0(long nativeObj);

    // C++:  void cv::dnn::Net::connect(String outPin, String inpPin)
    private static native void connect_0(long nativeObj, String outPin, String inpPin);

    // C++:  void cv::dnn::Net::dumpToFile(String path)
    private static native void dumpToFile_0(long nativeObj, String path);

    // C++:  void cv::dnn::Net::enableFusion(bool fusion)
    private static native void enableFusion_0(long nativeObj, boolean fusion);

    // C++:  void cv::dnn::Net::forward(vector_Mat& outputBlobs, String outputName = String())
    private static native void forward_2(long nativeObj, long outputBlobs_mat_nativeObj, String outputName);
    private static native void forward_3(long nativeObj, long outputBlobs_mat_nativeObj);

    // C++:  void cv::dnn::Net::forward(vector_Mat& outputBlobs, vector_String outBlobNames)
    private static native void forward_4(long nativeObj, long outputBlobs_mat_nativeObj, List<String> outBlobNames);

    // C++:  void cv::dnn::Net::getLayerTypes(vector_String& layersTypes)
    private static native void getLayerTypes_0(long nativeObj, List<String> layersTypes);

    // C++:  void cv::dnn::Net::getMemoryConsumption(MatShape netInputShape, size_t& weights, size_t& blobs)
    private static native void getMemoryConsumption_0(long nativeObj, long netInputShape_mat_nativeObj, double[] weights_out, double[] blobs_out);

    // C++:  void cv::dnn::Net::getMemoryConsumption(int layerId, MatShape netInputShape, size_t& weights, size_t& blobs)
    private static native void getMemoryConsumption_1(long nativeObj, int layerId, long netInputShape_mat_nativeObj, double[] weights_out, double[] blobs_out);

    // C++:  void cv::dnn::Net::getMemoryConsumption(int layerId, vector_MatShape netInputShapes, size_t& weights, size_t& blobs)
    private static native void getMemoryConsumption_2(long nativeObj, int layerId, List<MatOfInt> netInputShapes, double[] weights_out, double[] blobs_out);

    // C++:  void cv::dnn::Net::setHalideScheduler(String scheduler)
    private static native void setHalideScheduler_0(long nativeObj, String scheduler);

    // C++:  void cv::dnn::Net::setInput(Mat blob, String name = "", double scalefactor = 1.0, Scalar mean = Scalar())
    private static native void setInput_0(long nativeObj, long blob_nativeObj, String name, double scalefactor, double mean_val0, double mean_val1, double mean_val2, double mean_val3);
    private static native void setInput_1(long nativeObj, long blob_nativeObj, String name, double scalefactor);
    private static native void setInput_2(long nativeObj, long blob_nativeObj, String name);
    private static native void setInput_3(long nativeObj, long blob_nativeObj);

    // C++:  void cv::dnn::Net::setInputShape(String inputName, MatShape shape)
    private static native void setInputShape_0(long nativeObj, String inputName, long shape_mat_nativeObj);

    // C++:  void cv::dnn::Net::setInputsNames(vector_String inputBlobNames)
    private static native void setInputsNames_0(long nativeObj, List<String> inputBlobNames);

    // C++:  void cv::dnn::Net::setParam(LayerId layer, int numParam, Mat blob)
    private static native void setParam_0(long nativeObj, long layer_nativeObj, int numParam, long blob_nativeObj);

    // C++:  void cv::dnn::Net::setPreferableBackend(int backendId)
    private static native void setPreferableBackend_0(long nativeObj, int backendId);

    // C++:  void cv::dnn::Net::setPreferableTarget(int targetId)
    private static native void setPreferableTarget_0(long nativeObj, int targetId);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
