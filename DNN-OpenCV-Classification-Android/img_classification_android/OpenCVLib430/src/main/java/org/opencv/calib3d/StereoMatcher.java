//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.calib3d;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

// C++: class StereoMatcher
/**
 * The base class for stereo correspondence algorithms.
 */
public class StereoMatcher extends Algorithm {

    protected StereoMatcher(long addr) { super(addr); }

    // internal usage only
    public static StereoMatcher __fromPtr__(long addr) { return new StereoMatcher(addr); }

    // C++: enum <unnamed>
    public static final int
            DISP_SHIFT = 4,
            DISP_SCALE = (1 << DISP_SHIFT);


    //
    // C++:  int cv::StereoMatcher::getBlockSize()
    //

    public int getBlockSize() {
        return getBlockSize_0(nativeObj);
    }


    //
    // C++:  int cv::StereoMatcher::getDisp12MaxDiff()
    //

    public int getDisp12MaxDiff() {
        return getDisp12MaxDiff_0(nativeObj);
    }


    //
    // C++:  int cv::StereoMatcher::getMinDisparity()
    //

    public int getMinDisparity() {
        return getMinDisparity_0(nativeObj);
    }


    //
    // C++:  int cv::StereoMatcher::getNumDisparities()
    //

    public int getNumDisparities() {
        return getNumDisparities_0(nativeObj);
    }


    //
    // C++:  int cv::StereoMatcher::getSpeckleRange()
    //

    public int getSpeckleRange() {
        return getSpeckleRange_0(nativeObj);
    }


    //
    // C++:  int cv::StereoMatcher::getSpeckleWindowSize()
    //

    public int getSpeckleWindowSize() {
        return getSpeckleWindowSize_0(nativeObj);
    }


    //
    // C++:  void cv::StereoMatcher::compute(Mat left, Mat right, Mat& disparity)
    //

    /**
     * Computes disparity map for the specified stereo pair
     *
     *     @param left Left 8-bit single-channel image.
     *     @param right Right image of the same size and the same type as the left one.
     *     @param disparity Output disparity map. It has the same size as the input images. Some algorithms,
     *     like StereoBM or StereoSGBM compute 16-bit fixed-point disparity map (where each disparity value
     *     has 4 fractional bits), whereas other algorithms output 32-bit floating-point disparity map.
     */
    public void compute(Mat left, Mat right, Mat disparity) {
        compute_0(nativeObj, left.nativeObj, right.nativeObj, disparity.nativeObj);
    }


    //
    // C++:  void cv::StereoMatcher::setBlockSize(int blockSize)
    //

    public void setBlockSize(int blockSize) {
        setBlockSize_0(nativeObj, blockSize);
    }


    //
    // C++:  void cv::StereoMatcher::setDisp12MaxDiff(int disp12MaxDiff)
    //

    public void setDisp12MaxDiff(int disp12MaxDiff) {
        setDisp12MaxDiff_0(nativeObj, disp12MaxDiff);
    }


    //
    // C++:  void cv::StereoMatcher::setMinDisparity(int minDisparity)
    //

    public void setMinDisparity(int minDisparity) {
        setMinDisparity_0(nativeObj, minDisparity);
    }


    //
    // C++:  void cv::StereoMatcher::setNumDisparities(int numDisparities)
    //

    public void setNumDisparities(int numDisparities) {
        setNumDisparities_0(nativeObj, numDisparities);
    }


    //
    // C++:  void cv::StereoMatcher::setSpeckleRange(int speckleRange)
    //

    public void setSpeckleRange(int speckleRange) {
        setSpeckleRange_0(nativeObj, speckleRange);
    }


    //
    // C++:  void cv::StereoMatcher::setSpeckleWindowSize(int speckleWindowSize)
    //

    public void setSpeckleWindowSize(int speckleWindowSize) {
        setSpeckleWindowSize_0(nativeObj, speckleWindowSize);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  int cv::StereoMatcher::getBlockSize()
    private static native int getBlockSize_0(long nativeObj);

    // C++:  int cv::StereoMatcher::getDisp12MaxDiff()
    private static native int getDisp12MaxDiff_0(long nativeObj);

    // C++:  int cv::StereoMatcher::getMinDisparity()
    private static native int getMinDisparity_0(long nativeObj);

    // C++:  int cv::StereoMatcher::getNumDisparities()
    private static native int getNumDisparities_0(long nativeObj);

    // C++:  int cv::StereoMatcher::getSpeckleRange()
    private static native int getSpeckleRange_0(long nativeObj);

    // C++:  int cv::StereoMatcher::getSpeckleWindowSize()
    private static native int getSpeckleWindowSize_0(long nativeObj);

    // C++:  void cv::StereoMatcher::compute(Mat left, Mat right, Mat& disparity)
    private static native void compute_0(long nativeObj, long left_nativeObj, long right_nativeObj, long disparity_nativeObj);

    // C++:  void cv::StereoMatcher::setBlockSize(int blockSize)
    private static native void setBlockSize_0(long nativeObj, int blockSize);

    // C++:  void cv::StereoMatcher::setDisp12MaxDiff(int disp12MaxDiff)
    private static native void setDisp12MaxDiff_0(long nativeObj, int disp12MaxDiff);

    // C++:  void cv::StereoMatcher::setMinDisparity(int minDisparity)
    private static native void setMinDisparity_0(long nativeObj, int minDisparity);

    // C++:  void cv::StereoMatcher::setNumDisparities(int numDisparities)
    private static native void setNumDisparities_0(long nativeObj, int numDisparities);

    // C++:  void cv::StereoMatcher::setSpeckleRange(int speckleRange)
    private static native void setSpeckleRange_0(long nativeObj, int speckleRange);

    // C++:  void cv::StereoMatcher::setSpeckleWindowSize(int speckleWindowSize)
    private static native void setSpeckleWindowSize_0(long nativeObj, int speckleWindowSize);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
