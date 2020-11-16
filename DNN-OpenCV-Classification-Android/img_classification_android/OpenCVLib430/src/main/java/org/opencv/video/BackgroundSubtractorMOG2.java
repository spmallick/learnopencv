//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.core.Mat;
import org.opencv.video.BackgroundSubtractor;

// C++: class BackgroundSubtractorMOG2
/**
 * Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
 *
 * The class implements the Gaussian mixture model background subtraction described in CITE: Zivkovic2004
 * and CITE: Zivkovic2006 .
 */
public class BackgroundSubtractorMOG2 extends BackgroundSubtractor {

    protected BackgroundSubtractorMOG2(long addr) { super(addr); }

    // internal usage only
    public static BackgroundSubtractorMOG2 __fromPtr__(long addr) { return new BackgroundSubtractorMOG2(addr); }

    //
    // C++:  bool cv::BackgroundSubtractorMOG2::getDetectShadows()
    //

    /**
     * Returns the shadow detection flag
     *
     *     If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorMOG2 for
     *     details.
     * @return automatically generated
     */
    public boolean getDetectShadows() {
        return getDetectShadows_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorMOG2::getBackgroundRatio()
    //

    /**
     * Returns the "background ratio" parameter of the algorithm
     *
     *     If a foreground pixel keeps semi-constant value for about backgroundRatio\*history frames, it's
     *     considered background and added to the model as a center of a new component. It corresponds to TB
     *     parameter in the paper.
     * @return automatically generated
     */
    public double getBackgroundRatio() {
        return getBackgroundRatio_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorMOG2::getComplexityReductionThreshold()
    //

    /**
     * Returns the complexity reduction threshold
     *
     *     This parameter defines the number of samples needed to accept to prove the component exists. CT=0.05
     *     is a default value for all the samples. By setting CT=0 you get an algorithm very similar to the
     *     standard Stauffer&amp;Grimson algorithm.
     * @return automatically generated
     */
    public double getComplexityReductionThreshold() {
        return getComplexityReductionThreshold_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorMOG2::getShadowThreshold()
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
    // C++:  double cv::BackgroundSubtractorMOG2::getVarInit()
    //

    /**
     * Returns the initial variance of each gaussian component
     * @return automatically generated
     */
    public double getVarInit() {
        return getVarInit_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorMOG2::getVarMax()
    //

    public double getVarMax() {
        return getVarMax_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorMOG2::getVarMin()
    //

    public double getVarMin() {
        return getVarMin_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorMOG2::getVarThreshold()
    //

    /**
     * Returns the variance threshold for the pixel-model match
     *
     *     The main threshold on the squared Mahalanobis distance to decide if the sample is well described by
     *     the background model or not. Related to Cthr from the paper.
     * @return automatically generated
     */
    public double getVarThreshold() {
        return getVarThreshold_0(nativeObj);
    }


    //
    // C++:  double cv::BackgroundSubtractorMOG2::getVarThresholdGen()
    //

    /**
     * Returns the variance threshold for the pixel-model match used for new mixture component generation
     *
     *     Threshold for the squared Mahalanobis distance that helps decide when a sample is close to the
     *     existing components (corresponds to Tg in the paper). If a pixel is not close to any component, it
     *     is considered foreground or added as a new component. 3 sigma =&gt; Tg=3\*3=9 is default. A smaller Tg
     *     value generates more components. A higher Tg value may result in a small number of components but
     *     they can grow too large.
     * @return automatically generated
     */
    public double getVarThresholdGen() {
        return getVarThresholdGen_0(nativeObj);
    }


    //
    // C++:  int cv::BackgroundSubtractorMOG2::getHistory()
    //

    /**
     * Returns the number of last frames that affect the background model
     * @return automatically generated
     */
    public int getHistory() {
        return getHistory_0(nativeObj);
    }


    //
    // C++:  int cv::BackgroundSubtractorMOG2::getNMixtures()
    //

    /**
     * Returns the number of gaussian components in the background model
     * @return automatically generated
     */
    public int getNMixtures() {
        return getNMixtures_0(nativeObj);
    }


    //
    // C++:  int cv::BackgroundSubtractorMOG2::getShadowValue()
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
    // C++:  void cv::BackgroundSubtractorMOG2::apply(Mat image, Mat& fgmask, double learningRate = -1)
    //

    /**
     * Computes a foreground mask.
     *
     *     @param image Next video frame. Floating point frame will be used without scaling and should be in range \([0,255]\).
     *     @param fgmask The output foreground mask as an 8-bit binary image.
     *     @param learningRate The value between 0 and 1 that indicates how fast the background model is
     *     learnt. Negative parameter value makes the algorithm to use some automatically chosen learning
     *     rate. 0 means that the background model is not updated at all, 1 means that the background model
     *     is completely reinitialized from the last frame.
     */
    public void apply(Mat image, Mat fgmask, double learningRate) {
        apply_0(nativeObj, image.nativeObj, fgmask.nativeObj, learningRate);
    }

    /**
     * Computes a foreground mask.
     *
     *     @param image Next video frame. Floating point frame will be used without scaling and should be in range \([0,255]\).
     *     @param fgmask The output foreground mask as an 8-bit binary image.
     *     learnt. Negative parameter value makes the algorithm to use some automatically chosen learning
     *     rate. 0 means that the background model is not updated at all, 1 means that the background model
     *     is completely reinitialized from the last frame.
     */
    public void apply(Mat image, Mat fgmask) {
        apply_1(nativeObj, image.nativeObj, fgmask.nativeObj);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setBackgroundRatio(double ratio)
    //

    /**
     * Sets the "background ratio" parameter of the algorithm
     * @param ratio automatically generated
     */
    public void setBackgroundRatio(double ratio) {
        setBackgroundRatio_0(nativeObj, ratio);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setComplexityReductionThreshold(double ct)
    //

    /**
     * Sets the complexity reduction threshold
     * @param ct automatically generated
     */
    public void setComplexityReductionThreshold(double ct) {
        setComplexityReductionThreshold_0(nativeObj, ct);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setDetectShadows(bool detectShadows)
    //

    /**
     * Enables or disables shadow detection
     * @param detectShadows automatically generated
     */
    public void setDetectShadows(boolean detectShadows) {
        setDetectShadows_0(nativeObj, detectShadows);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setHistory(int history)
    //

    /**
     * Sets the number of last frames that affect the background model
     * @param history automatically generated
     */
    public void setHistory(int history) {
        setHistory_0(nativeObj, history);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setNMixtures(int nmixtures)
    //

    /**
     * Sets the number of gaussian components in the background model.
     *
     *     The model needs to be reinitalized to reserve memory.
     * @param nmixtures automatically generated
     */
    public void setNMixtures(int nmixtures) {
        setNMixtures_0(nativeObj, nmixtures);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setShadowThreshold(double threshold)
    //

    /**
     * Sets the shadow threshold
     * @param threshold automatically generated
     */
    public void setShadowThreshold(double threshold) {
        setShadowThreshold_0(nativeObj, threshold);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setShadowValue(int value)
    //

    /**
     * Sets the shadow value
     * @param value automatically generated
     */
    public void setShadowValue(int value) {
        setShadowValue_0(nativeObj, value);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setVarInit(double varInit)
    //

    /**
     * Sets the initial variance of each gaussian component
     * @param varInit automatically generated
     */
    public void setVarInit(double varInit) {
        setVarInit_0(nativeObj, varInit);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setVarMax(double varMax)
    //

    public void setVarMax(double varMax) {
        setVarMax_0(nativeObj, varMax);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setVarMin(double varMin)
    //

    public void setVarMin(double varMin) {
        setVarMin_0(nativeObj, varMin);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setVarThreshold(double varThreshold)
    //

    /**
     * Sets the variance threshold for the pixel-model match
     * @param varThreshold automatically generated
     */
    public void setVarThreshold(double varThreshold) {
        setVarThreshold_0(nativeObj, varThreshold);
    }


    //
    // C++:  void cv::BackgroundSubtractorMOG2::setVarThresholdGen(double varThresholdGen)
    //

    /**
     * Sets the variance threshold for the pixel-model match used for new mixture component generation
     * @param varThresholdGen automatically generated
     */
    public void setVarThresholdGen(double varThresholdGen) {
        setVarThresholdGen_0(nativeObj, varThresholdGen);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  bool cv::BackgroundSubtractorMOG2::getDetectShadows()
    private static native boolean getDetectShadows_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorMOG2::getBackgroundRatio()
    private static native double getBackgroundRatio_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorMOG2::getComplexityReductionThreshold()
    private static native double getComplexityReductionThreshold_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorMOG2::getShadowThreshold()
    private static native double getShadowThreshold_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorMOG2::getVarInit()
    private static native double getVarInit_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorMOG2::getVarMax()
    private static native double getVarMax_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorMOG2::getVarMin()
    private static native double getVarMin_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorMOG2::getVarThreshold()
    private static native double getVarThreshold_0(long nativeObj);

    // C++:  double cv::BackgroundSubtractorMOG2::getVarThresholdGen()
    private static native double getVarThresholdGen_0(long nativeObj);

    // C++:  int cv::BackgroundSubtractorMOG2::getHistory()
    private static native int getHistory_0(long nativeObj);

    // C++:  int cv::BackgroundSubtractorMOG2::getNMixtures()
    private static native int getNMixtures_0(long nativeObj);

    // C++:  int cv::BackgroundSubtractorMOG2::getShadowValue()
    private static native int getShadowValue_0(long nativeObj);

    // C++:  void cv::BackgroundSubtractorMOG2::apply(Mat image, Mat& fgmask, double learningRate = -1)
    private static native void apply_0(long nativeObj, long image_nativeObj, long fgmask_nativeObj, double learningRate);
    private static native void apply_1(long nativeObj, long image_nativeObj, long fgmask_nativeObj);

    // C++:  void cv::BackgroundSubtractorMOG2::setBackgroundRatio(double ratio)
    private static native void setBackgroundRatio_0(long nativeObj, double ratio);

    // C++:  void cv::BackgroundSubtractorMOG2::setComplexityReductionThreshold(double ct)
    private static native void setComplexityReductionThreshold_0(long nativeObj, double ct);

    // C++:  void cv::BackgroundSubtractorMOG2::setDetectShadows(bool detectShadows)
    private static native void setDetectShadows_0(long nativeObj, boolean detectShadows);

    // C++:  void cv::BackgroundSubtractorMOG2::setHistory(int history)
    private static native void setHistory_0(long nativeObj, int history);

    // C++:  void cv::BackgroundSubtractorMOG2::setNMixtures(int nmixtures)
    private static native void setNMixtures_0(long nativeObj, int nmixtures);

    // C++:  void cv::BackgroundSubtractorMOG2::setShadowThreshold(double threshold)
    private static native void setShadowThreshold_0(long nativeObj, double threshold);

    // C++:  void cv::BackgroundSubtractorMOG2::setShadowValue(int value)
    private static native void setShadowValue_0(long nativeObj, int value);

    // C++:  void cv::BackgroundSubtractorMOG2::setVarInit(double varInit)
    private static native void setVarInit_0(long nativeObj, double varInit);

    // C++:  void cv::BackgroundSubtractorMOG2::setVarMax(double varMax)
    private static native void setVarMax_0(long nativeObj, double varMax);

    // C++:  void cv::BackgroundSubtractorMOG2::setVarMin(double varMin)
    private static native void setVarMin_0(long nativeObj, double varMin);

    // C++:  void cv::BackgroundSubtractorMOG2::setVarThreshold(double varThreshold)
    private static native void setVarThreshold_0(long nativeObj, double varThreshold);

    // C++:  void cv::BackgroundSubtractorMOG2::setVarThresholdGen(double varThresholdGen)
    private static native void setVarThresholdGen_0(long nativeObj, double varThresholdGen);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
