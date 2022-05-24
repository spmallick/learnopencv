//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

// C++: class MergeExposures
/**
 * The base class algorithms that can merge exposure sequence to a single image.
 */
public class MergeExposures extends Algorithm {

    protected MergeExposures(long addr) { super(addr); }

    // internal usage only
    public static MergeExposures __fromPtr__(long addr) { return new MergeExposures(addr); }

    //
    // C++:  void cv::MergeExposures::process(vector_Mat src, Mat& dst, Mat times, Mat response)
    //

    /**
     * Merges images.
     *
     *     @param src vector of input images
     *     @param dst result image
     *     @param times vector of exposure time values for each image
     *     @param response 256x1 matrix with inverse camera response function for each pixel value, it should
     *     have the same number of channels as images.
     */
    public void process(List<Mat> src, Mat dst, Mat times, Mat response) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        process_0(nativeObj, src_mat.nativeObj, dst.nativeObj, times.nativeObj, response.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void cv::MergeExposures::process(vector_Mat src, Mat& dst, Mat times, Mat response)
    private static native void process_0(long nativeObj, long src_mat_nativeObj, long dst_nativeObj, long times_nativeObj, long response_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
