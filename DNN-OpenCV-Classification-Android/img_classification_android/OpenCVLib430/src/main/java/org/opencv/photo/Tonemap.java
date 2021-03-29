//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

// C++: class Tonemap
/**
 * Base class for tonemapping algorithms - tools that are used to map HDR image to 8-bit range.
 */
public class Tonemap extends Algorithm {

    protected Tonemap(long addr) { super(addr); }

    // internal usage only
    public static Tonemap __fromPtr__(long addr) { return new Tonemap(addr); }

    //
    // C++:  float cv::Tonemap::getGamma()
    //

    public float getGamma() {
        return getGamma_0(nativeObj);
    }


    //
    // C++:  void cv::Tonemap::process(Mat src, Mat& dst)
    //

    /**
     * Tonemaps image
     *
     *     @param src source image - CV_32FC3 Mat (float 32 bits 3 channels)
     *     @param dst destination image - CV_32FC3 Mat with values in [0, 1] range
     */
    public void process(Mat src, Mat dst) {
        process_0(nativeObj, src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::Tonemap::setGamma(float gamma)
    //

    public void setGamma(float gamma) {
        setGamma_0(nativeObj, gamma);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  float cv::Tonemap::getGamma()
    private static native float getGamma_0(long nativeObj);

    // C++:  void cv::Tonemap::process(Mat src, Mat& dst)
    private static native void process_0(long nativeObj, long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::Tonemap::setGamma(float gamma)
    private static native void setGamma_0(long nativeObj, float gamma);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
