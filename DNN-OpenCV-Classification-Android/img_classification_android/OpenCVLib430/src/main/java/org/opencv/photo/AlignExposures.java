//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

// C++: class AlignExposures
/**
 * The base class for algorithms that align images of the same scene with different exposures
 */
public class AlignExposures extends Algorithm {

    protected AlignExposures(long addr) { super(addr); }

    // internal usage only
    public static AlignExposures __fromPtr__(long addr) { return new AlignExposures(addr); }

    //
    // C++:  void cv::AlignExposures::process(vector_Mat src, vector_Mat dst, Mat times, Mat response)
    //

    /**
     * Aligns images
     *
     *     @param src vector of input images
     *     @param dst vector of aligned images
     *     @param times vector of exposure time values for each image
     *     @param response 256x1 matrix with inverse camera response function for each pixel value, it should
     *     have the same number of channels as images.
     */
    public void process(List<Mat> src, List<Mat> dst, Mat times, Mat response) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        Mat dst_mat = Converters.vector_Mat_to_Mat(dst);
        process_0(nativeObj, src_mat.nativeObj, dst_mat.nativeObj, times.nativeObj, response.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void cv::AlignExposures::process(vector_Mat src, vector_Mat dst, Mat times, Mat response)
    private static native void process_0(long nativeObj, long src_mat_nativeObj, long dst_mat_nativeObj, long times_nativeObj, long response_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
