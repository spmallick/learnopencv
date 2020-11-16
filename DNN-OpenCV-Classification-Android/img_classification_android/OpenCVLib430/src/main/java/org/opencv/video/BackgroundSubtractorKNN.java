//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.video.BackgroundSubtractor;

// C++: class BackgroundSubtractorKNN
/**
 * K-nearest neighbours - based Background/Foreground Segmentation Algorithm.
 *
 * The class implements the K-nearest neighbours background subtraction described in CITE: Zivkovic2006 .
 * Very efficient if number of foreground pixels is low.
 */
public class BackgroundSubtractorKNN extends BackgroundSubtractor {

    protected BackgroundSubtractorKNN(long addr) { super(addr); }

    // internal usage only
    public static BackgroundSubtractorKNN __fromPtr__(long addr) { return new BackgroundSubtractorKNN(addr); }

    //
    // C++:  bool cv::BackgroundSubtractorKNN::getDetectShadows()
    //

    /**
     * Returns the shadow detection flag
     *
     *     If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorKNN for
     *     details.
     * @return automatically generated
     */
    public boolean getDetectShadows() {
        return getDetectShadows_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorKNN::getDist2Threshold()
    //

    /**
     * Returns the threshold on the squared distance between the pixel and the sample
     *
     *     The threshold on the squared distance between the pixel and the sample to decide whether a pixel is
     *     close to a data sample.
     * @return automatically generated
     */
    public double getDist2Threshold() {
        return getDist2Threshold_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorKNN::getShadowThreshold()
    //

    /**
     * Returns the shadow threshold
     *
     *     A shadow is detected if pixel is a darker version of the background. The shadow threshold (Tau in
     *     the paper) is a threshold defining how much darker the shadow can be. Tau= 0.5 means that if a pixel
     *     is more than twice darker then it is not shadow. See Prati, Mikic, Trivedi and Cucchiara,
     * Detecting Moving Shadows...*, IEEE PAMI,2003.
     * @return automatically generated
     */
    public double getShadowThreshold() {
        return getShadowThreshold_0(nativeObj);
    }


    //
    // C++:  int cv::BackgroundSubtractorKNN::getHistory()
    //

    /**
     * Returns the number of last frames that affect the background model
     * @return automatically generated
     */
    public int getHistory() {
        return getHistory_0(nativeObj);
    }


    //
    // C++:  int cv::BackgroundSubtractorKNN::getNSamples()
    //

    /**
     * Returns the number of data samples in the background model
     * @return automatically generated
     */
    public int getNSamples() {
        return getNSamples_0(nativeObj);
    }


    //
    // C++:  int cv::BackgroundSubtractorKNN::getShadowValue()
    //

    /**
     * Returns the shadow value
     *
     *     Shadow value is the value used to mark shadows in the foreground mask. Default value is 127. Value 0
     *     in the mask always means background, 255 means foreground.
     * @return automatically generated
     */
    public int getShadowValue() {
        return getShadowValue_0(nativeObj);
    }


    //
    // C++:  int cv::BackgroundSubtractorKNN::getkNNSamples()
    //

    /**
     * Returns the number of neighbours, the k in the kNN.
     *
     *     K is the number of samples that need to be within dist2Threshold in order to decide that that
     *     pixel is matching the kNN background model.
     * @return automatically generated
     */
    public int getkNNSamples() {
        return getkNNSamples_0(nativeObj);
    }


    //
    // C++:  void cv::BackgroundSubtractorKNN::setDetectShadows(bool detectShadows)
    //

    /**
     * Enables or disables shadow detection
     * @param detectShadows automatically generated
     */
    public void setDetectShadows(boolean detectShadows) {
        setDetectShadows_0(nativeObj, detectShadows);
    }


    //
    // C++:  void cv::BackgroundSubtractorKNN::setDist2Threshold(double _dist2Threshold)
    //

    /**
     * Sets the threshold on the squared distance
     * @param _dist2Threshold automatically generated
     */
    public void setDist2Threshold(double _dist2Threshold) {
        setDist2Threshold_0(nativeObj, _dist2Threshold);
    }


    //
    // C++:  void cv::BackgroundSubtractorKNN::setHistory(int history)
    //

    /**
     * Sets the number of last frames that affect the background model
     * @param history automatically generated
     */
    public void setHistory(int history) {
        setHistory_0(nativeObj, history);
    }


    //
    // C++:  void cv::BackgroundSubtractorKNN::setNSamples(int _nN)
    //

    /**
     * Sets the number of data samples in the background model.
     *
     *     The model needs to be reinitalized to reserve memory.
     * @param _nN automatically generated
     */
    public void setNSamples(int _nN) {
        setNSamples_0(nativeObj, _nN);
    }


    //
    // C++:  void cv::BackgroundSubtractorKNN::setShadowThreshold(double threshold)
    //

    /**
     * Sets the shadow threshold
     * @param threshold automatically generated
     */
    public void setShadowThreshold(double threshold) {
        setShadowThreshold_0(nativeObj, threshold);
    }


    //
    // C++:  void cv::BackgroundSubtractorKNN::setShadowValue(int value)
    //

    /**
     * Sets the shadow value
     * @param value automatically generated
     */
    public void setShadowValue(int value) {
        setShadowValue_0(nativeObj, value);
    }


    //
    // C++:  void cv::BackgroundSubtractorKNN::setkNNSamples(int _nkNN)
    //

    /**
     * Sets the k in the kNN. How many nearest neighbours need to match.
     * @param _nkNN automatically generated
     */
    public void setkNNSamples(int _nkNN) {
        setkNNSamples_0(nativeObj, _nkNN);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  bool cv::BackgroundSubtractorKNN::getDetectShadows()
    private static native boolean getDetectShadows_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorKNN::getDist2Threshold()
    private static native double getDist2Threshold_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorKNN::getShadowThreshold()
    private static native double getShadowThreshold_0(long nativeObj);

    // C++:  int cv::BackgroundSubtractorKNN::getHistory()
    private static native int getHistory_0(long nativeObj);

    // C++:  int cv::BackgroundSubtractorKNN::getNSamples()
    private static native int getNSamples_0(long nativeObj);

    // C++:  int cv::BackgroundSubtractorKNN::getShadowValue()
    private static native int getShadowValue_0(long nativeObj);

    // C++:  int cv::BackgroundSubtractorKNN::getkNNSamples()
    private static native int getkNNSamples_0(long nativeObj);

    // C++:  void cv::BackgroundSubtractorKNN::setDetectShadows(bool detectShadows)
    private static native void setDetectShadows_0(long nativeObj, boolean detectShadows);

    // C++:  void cv::BackgroundSubtractorKNN::setDist2Threshold(double _dist2Threshold)
    private static native void setDist2Threshold_0(long nativeObj, double _dist2Threshold);

    // C++:  void cv::BackgroundSubtractorKNN::setHistory(int history)
    private static native void setHistory_0(long nativeObj, int history);

    // C++:  void cv::BackgroundSubtractorKNN::setNSamples(int _nN)
    private static native void setNSamples_0(long nativeObj, int _nN);

    // C++:  void cv::BackgroundSubtractorKNN::setShadowThreshold(double threshold)
    private static native void setShadowThreshold_0(long nativeObj, double threshold);

    // C++:  void cv::BackgroundSubtractorKNN::setShadowValue(int value)
    private static native void setShadowValue_0(long nativeObj, int value);

    // C++:  void cv::BackgroundSubtractorKNN::setkNNSamples(int _nkNN)
    private static native void setkNNSamples_0(long nativeObj, int _nkNN);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
