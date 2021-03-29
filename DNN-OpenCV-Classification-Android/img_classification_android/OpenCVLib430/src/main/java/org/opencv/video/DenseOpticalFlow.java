//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

// C++: class DenseOpticalFlow
/**
 * Base class for dense optical flow algorithms
 */
public class DenseOpticalFlow extends Algorithm {

    protected DenseOpticalFlow(long addr) { super(addr); }

    // internal usage only
    public static DenseOpticalFlow __fromPtr__(long addr) { return new DenseOpticalFlow(addr); }

    //
    // C++:  void cv::DenseOpticalFlow::calc(Mat I0, Mat I1, Mat& flow)
    //

    /**
     * Calculates an optical flow.
     *
     *     @param I0 first 8-bit single-channel input image.
     *     @param I1 second input image of the same size and the same type as prev.
     *     @param flow computed flow image that has the same size as prev and type CV_32FC2.
     */
    public void calc(Mat I0, Mat I1, Mat flow) {
        calc_0(nativeObj, I0.nativeObj, I1.nativeObj, flow.nativeObj);
    }


    //
    // C++:  void cv::DenseOpticalFlow::collectGarbage()
    //

    /**
     * Releases all inner buffers.
     */
    public void collectGarbage() {
        collectGarbage_0(nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void cv::DenseOpticalFlow::calc(Mat I0, Mat I1, Mat& flow)
    private static native void calc_0(long nativeObj, long I0_nativeObj, long I1_nativeObj, long flow_nativeObj);

    // C++:  void cv::DenseOpticalFlow::collectGarbage()
    private static native void collectGarbage_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
