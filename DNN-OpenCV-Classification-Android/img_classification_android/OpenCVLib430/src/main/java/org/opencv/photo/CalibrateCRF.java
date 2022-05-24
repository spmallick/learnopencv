//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

// C++: class CalibrateCRF
/**
 * The base class for camera response calibration algorithms.
 */
public class CalibrateCRF extends Algorithm {

    protected CalibrateCRF(long addr) { super(addr); }

    // internal usage only
    public static CalibrateCRF __fromPtr__(long addr) { return new CalibrateCRF(addr); }

    //
    // C++:  void cv::CalibrateCRF::process(vector_Mat src, Mat& dst, Mat times)
    //

    /**
     * Recovers inverse camera response.
     *
     *     @param src vector of input images
     *     @param dst 256x1 matrix with inverse camera response function
     *     @param times vector of exposure time values for each image
     */
    public void process(List<Mat> src, Mat dst, Mat times) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        process_0(nativeObj, src_mat.nativeObj, dst.nativeObj, times.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void cv::CalibrateCRF::process(vector_Mat src, Mat& dst, Mat times)
    private static native void process_0(long nativeObj, long src_mat_nativeObj, long dst_nativeObj, long times_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
