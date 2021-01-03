//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.photo.MergeExposures;
import org.opencv.utils.Converters;

// C++: class MergeMertens
/**
 * Pixels are weighted using contrast, saturation and well-exposedness measures, than images are
 * combined using laplacian pyramids.
 *
 * The resulting image weight is constructed as weighted average of contrast, saturation and
 * well-exposedness measures.
 *
 * The resulting image doesn't require tonemapping and can be converted to 8-bit image by multiplying
 * by 255, but it's recommended to apply gamma correction and/or linear tonemapping.
 *
 * For more information see CITE: MK07 .
 */
public class MergeMertens extends MergeExposures {

    protected MergeMertens(long addr) { super(addr); }

    // internal usage only
    public static MergeMertens __fromPtr__(long addr) { return new MergeMertens(addr); }

    //
    // C++:  float cv::MergeMertens::getContrastWeight()
    //

    public float getContrastWeight() {
        return getContrastWeight_0(nativeObj);
    }


    //
    // C++:  float cv::MergeMertens::getExposureWeight()
    //

    public float getExposureWeight() {
        return getExposureWeight_0(nativeObj);
    }


    //
    // C++:  float cv::MergeMertens::getSaturationWeight()
    //

    public float getSaturationWeight() {
        return getSaturationWeight_0(nativeObj);
    }


    //
    // C++:  void cv::MergeMertens::process(vector_Mat src, Mat& dst, Mat times, Mat response)
    //

    public void process(List<Mat> src, Mat dst, Mat times, Mat response) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        process_0(nativeObj, src_mat.nativeObj, dst.nativeObj, times.nativeObj, response.nativeObj);
    }


    //
    // C++:  void cv::MergeMertens::process(vector_Mat src, Mat& dst)
    //

    /**
     * Short version of process, that doesn't take extra arguments.
     *
     *     @param src vector of input images
     *     @param dst result image
     */
    public void process(List<Mat> src, Mat dst) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        process_1(nativeObj, src_mat.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::MergeMertens::setContrastWeight(float contrast_weiht)
    //

    public void setContrastWeight(float contrast_weiht) {
        setContrastWeight_0(nativeObj, contrast_weiht);
    }


    //
    // C++:  void cv::MergeMertens::setExposureWeight(float exposure_weight)
    //

    public void setExposureWeight(float exposure_weight) {
        setExposureWeight_0(nativeObj, exposure_weight);
    }


    //
    // C++:  void cv::MergeMertens::setSaturationWeight(float saturation_weight)
    //

    public void setSaturationWeight(float saturation_weight) {
        setSaturationWeight_0(nativeObj, saturation_weight);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  float cv::MergeMertens::getContrastWeight()
    private static native float getContrastWeight_0(long nativeObj);

    // C++:  float cv::MergeMertens::getExposureWeight()
    private static native float getExposureWeight_0(long nativeObj);

    // C++:  float cv::MergeMertens::getSaturationWeight()
    private static native float getSaturationWeight_0(long nativeObj);

    // C++:  void cv::MergeMertens::process(vector_Mat src, Mat& dst, Mat times, Mat response)
    private static native void process_0(long nativeObj, long src_mat_nativeObj, long dst_nativeObj, long times_nativeObj, long response_nativeObj);

    // C++:  void cv::MergeMertens::process(vector_Mat src, Mat& dst)
    private static native void process_1(long nativeObj, long src_mat_nativeObj, long dst_nativeObj);

    // C++:  void cv::MergeMertens::setContrastWeight(float contrast_weiht)
    private static native void setContrastWeight_0(long nativeObj, float contrast_weiht);

    // C++:  void cv::MergeMertens::setExposureWeight(float exposure_weight)
    private static native void setExposureWeight_0(long nativeObj, float exposure_weight);

    // C++:  void cv::MergeMertens::setSaturationWeight(float saturation_weight)
    private static native void setSaturationWeight_0(long nativeObj, float saturation_weight);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
