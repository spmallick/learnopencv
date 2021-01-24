//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;



// C++: class Params

public class Params {

    protected final long nativeObj;
    protected Params(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static Params __fromPtr__(long addr) { return new Params(addr); }

    //
    // C++:   cv::SimpleBlobDetector::Params::Params()
    //

    public Params() {
        nativeObj = Params_0();
    }


    //
    // C++: float Params::thresholdStep
    //

    public float get_thresholdStep() {
        return get_thresholdStep_0(nativeObj);
    }


    //
    // C++: void Params::thresholdStep
    //

    public void set_thresholdStep(float thresholdStep) {
        set_thresholdStep_0(nativeObj, thresholdStep);
    }


    //
    // C++: float Params::minThreshold
    //

    public float get_minThreshold() {
        return get_minThreshold_0(nativeObj);
    }


    //
    // C++: void Params::minThreshold
    //

    public void set_minThreshold(float minThreshold) {
        set_minThreshold_0(nativeObj, minThreshold);
    }


    //
    // C++: float Params::maxThreshold
    //

    public float get_maxThreshold() {
        return get_maxThreshold_0(nativeObj);
    }


    //
    // C++: void Params::maxThreshold
    //

    public void set_maxThreshold(float maxThreshold) {
        set_maxThreshold_0(nativeObj, maxThreshold);
    }


    //
    // C++: size_t Params::minRepeatability
    //

    public long get_minRepeatability() {
        return get_minRepeatability_0(nativeObj);
    }


    //
    // C++: void Params::minRepeatability
    //

    public void set_minRepeatability(long minRepeatability) {
        set_minRepeatability_0(nativeObj, minRepeatability);
    }


    //
    // C++: float Params::minDistBetweenBlobs
    //

    public float get_minDistBetweenBlobs() {
        return get_minDistBetweenBlobs_0(nativeObj);
    }


    //
    // C++: void Params::minDistBetweenBlobs
    //

    public void set_minDistBetweenBlobs(float minDistBetweenBlobs) {
        set_minDistBetweenBlobs_0(nativeObj, minDistBetweenBlobs);
    }


    //
    // C++: bool Params::filterByColor
    //

    public boolean get_filterByColor() {
        return get_filterByColor_0(nativeObj);
    }


    //
    // C++: void Params::filterByColor
    //

    public void set_filterByColor(boolean filterByColor) {
        set_filterByColor_0(nativeObj, filterByColor);
    }


    //
    // C++: uchar Params::blobColor
    //

    // Return type 'uchar' is not supported, skipping the function


    //
    // C++: void Params::blobColor
    //

    // Unknown type 'uchar' (I), skipping the function


    //
    // C++: bool Params::filterByArea
    //

    public boolean get_filterByArea() {
        return get_filterByArea_0(nativeObj);
    }


    //
    // C++: void Params::filterByArea
    //

    public void set_filterByArea(boolean filterByArea) {
        set_filterByArea_0(nativeObj, filterByArea);
    }


    //
    // C++: float Params::minArea
    //

    public float get_minArea() {
        return get_minArea_0(nativeObj);
    }


    //
    // C++: void Params::minArea
    //

    public void set_minArea(float minArea) {
        set_minArea_0(nativeObj, minArea);
    }


    //
    // C++: float Params::maxArea
    //

    public float get_maxArea() {
        return get_maxArea_0(nativeObj);
    }


    //
    // C++: void Params::maxArea
    //

    public void set_maxArea(float maxArea) {
        set_maxArea_0(nativeObj, maxArea);
    }


    //
    // C++: bool Params::filterByCircularity
    //

    public boolean get_filterByCircularity() {
        return get_filterByCircularity_0(nativeObj);
    }


    //
    // C++: void Params::filterByCircularity
    //

    public void set_filterByCircularity(boolean filterByCircularity) {
        set_filterByCircularity_0(nativeObj, filterByCircularity);
    }


    //
    // C++: float Params::minCircularity
    //

    public float get_minCircularity() {
        return get_minCircularity_0(nativeObj);
    }


    //
    // C++: void Params::minCircularity
    //

    public void set_minCircularity(float minCircularity) {
        set_minCircularity_0(nativeObj, minCircularity);
    }


    //
    // C++: float Params::maxCircularity
    //

    public float get_maxCircularity() {
        return get_maxCircularity_0(nativeObj);
    }


    //
    // C++: void Params::maxCircularity
    //

    public void set_maxCircularity(float maxCircularity) {
        set_maxCircularity_0(nativeObj, maxCircularity);
    }


    //
    // C++: bool Params::filterByInertia
    //

    public boolean get_filterByInertia() {
        return get_filterByInertia_0(nativeObj);
    }


    //
    // C++: void Params::filterByInertia
    //

    public void set_filterByInertia(boolean filterByInertia) {
        set_filterByInertia_0(nativeObj, filterByInertia);
    }


    //
    // C++: float Params::minInertiaRatio
    //

    public float get_minInertiaRatio() {
        return get_minInertiaRatio_0(nativeObj);
    }


    //
    // C++: void Params::minInertiaRatio
    //

    public void set_minInertiaRatio(float minInertiaRatio) {
        set_minInertiaRatio_0(nativeObj, minInertiaRatio);
    }


    //
    // C++: float Params::maxInertiaRatio
    //

    public float get_maxInertiaRatio() {
        return get_maxInertiaRatio_0(nativeObj);
    }


    //
    // C++: void Params::maxInertiaRatio
    //

    public void set_maxInertiaRatio(float maxInertiaRatio) {
        set_maxInertiaRatio_0(nativeObj, maxInertiaRatio);
    }


    //
    // C++: bool Params::filterByConvexity
    //

    public boolean get_filterByConvexity() {
        return get_filterByConvexity_0(nativeObj);
    }


    //
    // C++: void Params::filterByConvexity
    //

    public void set_filterByConvexity(boolean filterByConvexity) {
        set_filterByConvexity_0(nativeObj, filterByConvexity);
    }


    //
    // C++: float Params::minConvexity
    //

    public float get_minConvexity() {
        return get_minConvexity_0(nativeObj);
    }


    //
    // C++: void Params::minConvexity
    //

    public void set_minConvexity(float minConvexity) {
        set_minConvexity_0(nativeObj, minConvexity);
    }


    //
    // C++: float Params::maxConvexity
    //

    public float get_maxConvexity() {
        return get_maxConvexity_0(nativeObj);
    }


    //
    // C++: void Params::maxConvexity
    //

    public void set_maxConvexity(float maxConvexity) {
        set_maxConvexity_0(nativeObj, maxConvexity);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::SimpleBlobDetector::Params::Params()
    private static native long Params_0();

    // C++: float Params::thresholdStep
    private static native float get_thresholdStep_0(long nativeObj);

    // C++: void Params::thresholdStep
    private static native void set_thresholdStep_0(long nativeObj, float thresholdStep);

    // C++: float Params::minThreshold
    private static native float get_minThreshold_0(long nativeObj);

    // C++: void Params::minThreshold
    private static native void set_minThreshold_0(long nativeObj, float minThreshold);

    // C++: float Params::maxThreshold
    private static native float get_maxThreshold_0(long nativeObj);

    // C++: void Params::maxThreshold
    private static native void set_maxThreshold_0(long nativeObj, float maxThreshold);

    // C++: size_t Params::minRepeatability
    private static native long get_minRepeatability_0(long nativeObj);

    // C++: void Params::minRepeatability
    private static native void set_minRepeatability_0(long nativeObj, long minRepeatability);

    // C++: float Params::minDistBetweenBlobs
    private static native float get_minDistBetweenBlobs_0(long nativeObj);

    // C++: void Params::minDistBetweenBlobs
    private static native void set_minDistBetweenBlobs_0(long nativeObj, float minDistBetweenBlobs);

    // C++: bool Params::filterByColor
    private static native boolean get_filterByColor_0(long nativeObj);

    // C++: void Params::filterByColor
    private static native void set_filterByColor_0(long nativeObj, boolean filterByColor);

    // C++: bool Params::filterByArea
    private static native boolean get_filterByArea_0(long nativeObj);

    // C++: void Params::filterByArea
    private static native void set_filterByArea_0(long nativeObj, boolean filterByArea);

    // C++: float Params::minArea
    private static native float get_minArea_0(long nativeObj);

    // C++: void Params::minArea
    private static native void set_minArea_0(long nativeObj, float minArea);

    // C++: float Params::maxArea
    private static native float get_maxArea_0(long nativeObj);

    // C++: void Params::maxArea
    private static native void set_maxArea_0(long nativeObj, float maxArea);

    // C++: bool Params::filterByCircularity
    private static native boolean get_filterByCircularity_0(long nativeObj);

    // C++: void Params::filterByCircularity
    private static native void set_filterByCircularity_0(long nativeObj, boolean filterByCircularity);

    // C++: float Params::minCircularity
    private static native float get_minCircularity_0(long nativeObj);

    // C++: void Params::minCircularity
    private static native void set_minCircularity_0(long nativeObj, float minCircularity);

    // C++: float Params::maxCircularity
    private static native float get_maxCircularity_0(long nativeObj);

    // C++: void Params::maxCircularity
    private static native void set_maxCircularity_0(long nativeObj, float maxCircularity);

    // C++: bool Params::filterByInertia
    private static native boolean get_filterByInertia_0(long nativeObj);

    // C++: void Params::filterByInertia
    private static native void set_filterByInertia_0(long nativeObj, boolean filterByInertia);

    // C++: float Params::minInertiaRatio
    private static native float get_minInertiaRatio_0(long nativeObj);

    // C++: void Params::minInertiaRatio
    private static native void set_minInertiaRatio_0(long nativeObj, float minInertiaRatio);

    // C++: float Params::maxInertiaRatio
    private static native float get_maxInertiaRatio_0(long nativeObj);

    // C++: void Params::maxInertiaRatio
    private static native void set_maxInertiaRatio_0(long nativeObj, float maxInertiaRatio);

    // C++: bool Params::filterByConvexity
    private static native boolean get_filterByConvexity_0(long nativeObj);

    // C++: void Params::filterByConvexity
    private static native void set_filterByConvexity_0(long nativeObj, boolean filterByConvexity);

    // C++: float Params::minConvexity
    private static native float get_minConvexity_0(long nativeObj);

    // C++: void Params::minConvexity
    private static native void set_minConvexity_0(long nativeObj, float minConvexity);

    // C++: float Params::maxConvexity
    private static native float get_maxConvexity_0(long nativeObj);

    // C++: void Params::maxConvexity
    private static native void set_maxConvexity_0(long nativeObj, float maxConvexity);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
