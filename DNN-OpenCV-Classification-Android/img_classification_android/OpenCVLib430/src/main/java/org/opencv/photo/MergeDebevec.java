//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.photo.MergeExposures;
import org.opencv.utils.Converters;

// C++: class MergeDebevec
/**
 * The resulting HDR image is calculated as weighted average of the exposures considering exposure
 * values and camera response.
 *
 * For more information see CITE: DM97 .
 */
public class MergeDebevec extends MergeExposures {

    protected MergeDebevec(long addr) { super(addr); }

    // internal usage only
    public static MergeDebevec __fromPtr__(long addr) { return new MergeDebevec(addr); }

    //
    // C++:  void cv::MergeDebevec::process(vector_Mat src, Mat& dst, Mat times, Mat response)
    //

    public void process(List<Mat> src, Mat dst, Mat times, Mat response) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        process_0(nativeObj, src_mat.nativeObj, dst.nativeObj, times.nativeObj, response.nativeObj);
    }


    //
    // C++:  void cv::MergeDebevec::process(vector_Mat src, Mat& dst, Mat times)
    //

    public void process(List<Mat> src, Mat dst, Mat times) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        process_1(nativeObj, src_mat.nativeObj, dst.nativeObj, times.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void cv::MergeDebevec::process(vector_Mat src, Mat& dst, Mat times, Mat response)
    private static native void process_0(long nativeObj, long src_mat_nativeObj, long dst_nativeObj, long times_nativeObj, long response_nativeObj);

    // C++:  void cv::MergeDebevec::process(vector_Mat src, Mat& dst, Mat times)
    private static native void process_1(long nativeObj, long src_mat_nativeObj, long dst_nativeObj, long times_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
