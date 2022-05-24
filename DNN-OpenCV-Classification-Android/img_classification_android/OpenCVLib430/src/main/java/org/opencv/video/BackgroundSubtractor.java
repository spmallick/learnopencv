//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

// C++: class BackgroundSubtractor
/**
 * Base class for background/foreground segmentation. :
 *
 * The class is only used to define the common interface for the whole family of background/foreground
 * segmentation algorithms.
 */
public class BackgroundSubtractor extends Algorithm {

    protected BackgroundSubtractor(long addr) { super(addr); }

    // internal usage only
    public static BackgroundSubtractor __fromPtr__(long addr) { return new BackgroundSubtractor(addr); }

    //
    // C++:  void cv::BackgroundSubtractor::apply(Mat image, Mat& fgmask, double learningRate = -1)
    //

    /**
     * Computes a foreground mask.
     *
     *     @param image Next video frame.
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
     *     @param image Next video frame.
     *     @param fgmask The output foreground mask as an 8-bit binary image.
     *     learnt. Negative parameter value makes the algorithm to use some automatically chosen learning
     *     rate. 0 means that the background model is not updated at all, 1 means that the background model
     *     is completely reinitialized from the last frame.
     */
    public void apply(Mat image, Mat fgmask) {
        apply_1(nativeObj, image.nativeObj, fgmask.nativeObj);
    }


    //
    // C++:  void cv::BackgroundSubtractor::getBackgroundImage(Mat& backgroundImage)
    //

    /**
     * Computes a background image.
     *
     *     @param backgroundImage The output background image.
     *
     *     <b>Note:</b> Sometimes the background image can be very blurry, as it contain the average background
     *     statistics.
     */
    public void getBackgroundImage(Mat backgroundImage) {
        getBackgroundImage_0(nativeObj, backgroundImage.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void cv::BackgroundSubtractor::apply(Mat image, Mat& fgmask, double learningRate = -1)
    private static native void apply_0(long nativeObj, long image_nativeObj, long fgmask_nativeObj, double learningRate);
    private static native void apply_1(long nativeObj, long image_nativeObj, long fgmask_nativeObj);

    // C++:  void cv::BackgroundSubtractor::getBackgroundImage(Mat& backgroundImage)
    private static native void getBackgroundImage_0(long nativeObj, long backgroundImage_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
