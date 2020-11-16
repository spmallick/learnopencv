//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.photo.AlignExposures;
import org.opencv.utils.Converters;

// C++: class AlignMTB
/**
 * This algorithm converts images to median threshold bitmaps (1 for pixels brighter than median
 * luminance and 0 otherwise) and than aligns the resulting bitmaps using bit operations.
 *
 * It is invariant to exposure, so exposure values and camera response are not necessary.
 *
 * In this implementation new image regions are filled with zeros.
 *
 * For more information see CITE: GW03 .
 */
public class AlignMTB extends AlignExposures {

    protected AlignMTB(long addr) { super(addr); }

    // internal usage only
    public static AlignMTB __fromPtr__(long addr) { return new AlignMTB(addr); }

    //
    // C++:  Point cv::AlignMTB::calculateShift(Mat img0, Mat img1)
    //

    /**
     * Calculates shift between two images, i. e. how to shift the second image to correspond it with the
     *     first.
     *
     *     @param img0 first image
     *     @param img1 second image
     * @return automatically generated
     */
    public Point calculateShift(Mat img0, Mat img1) {
        return new Point(calculateShift_0(nativeObj, img0.nativeObj, img1.nativeObj));
    }


    //
    // C++:  bool cv::AlignMTB::getCut()
    //

    public boolean getCut() {
        return getCut_0(nativeObj);
    }


    //
    // C++:  int cv::AlignMTB::getExcludeRange()
    //

    public int getExcludeRange() {
        return getExcludeRange_0(nativeObj);
    }


    //
    // C++:  int cv::AlignMTB::getMaxBits()
    //

    public int getMaxBits() {
        return getMaxBits_0(nativeObj);
    }


    //
    // C++:  void cv::AlignMTB::computeBitmaps(Mat img, Mat& tb, Mat& eb)
    //

    /**
     * Computes median threshold and exclude bitmaps of given image.
     *
     *     @param img input image
     *     @param tb median threshold bitmap
     *     @param eb exclude bitmap
     */
    public void computeBitmaps(Mat img, Mat tb, Mat eb) {
        computeBitmaps_0(nativeObj, img.nativeObj, tb.nativeObj, eb.nativeObj);
    }


    //
    // C++:  void cv::AlignMTB::process(vector_Mat src, vector_Mat dst, Mat times, Mat response)
    //

    public void process(List<Mat> src, List<Mat> dst, Mat times, Mat response) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        Mat dst_mat = Converters.vector_Mat_to_Mat(dst);
        process_0(nativeObj, src_mat.nativeObj, dst_mat.nativeObj, times.nativeObj, response.nativeObj);
    }


    //
    // C++:  void cv::AlignMTB::process(vector_Mat src, vector_Mat dst)
    //

    /**
     * Short version of process, that doesn't take extra arguments.
     *
     *     @param src vector of input images
     *     @param dst vector of aligned images
     */
    public void process(List<Mat> src, List<Mat> dst) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        Mat dst_mat = Converters.vector_Mat_to_Mat(dst);
        process_1(nativeObj, src_mat.nativeObj, dst_mat.nativeObj);
    }


    //
    // C++:  void cv::AlignMTB::setCut(bool value)
    //

    public void setCut(boolean value) {
        setCut_0(nativeObj, value);
    }


    //
    // C++:  void cv::AlignMTB::setExcludeRange(int exclude_range)
    //

    public void setExcludeRange(int exclude_range) {
        setExcludeRange_0(nativeObj, exclude_range);
    }


    //
    // C++:  void cv::AlignMTB::setMaxBits(int max_bits)
    //

    public void setMaxBits(int max_bits) {
        setMaxBits_0(nativeObj, max_bits);
    }


    //
    // C++:  void cv::AlignMTB::shiftMat(Mat src, Mat& dst, Point shift)
    //

    /**
     * Helper function, that shift Mat filling new regions with zeros.
     *
     *     @param src input image
     *     @param dst result image
     *     @param shift shift value
     */
    public void shiftMat(Mat src, Mat dst, Point shift) {
        shiftMat_0(nativeObj, src.nativeObj, dst.nativeObj, shift.x, shift.y);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Point cv::AlignMTB::calculateShift(Mat img0, Mat img1)
    private static native double[] calculateShift_0(long nativeObj, long img0_nativeObj, long img1_nativeObj);

    // C++:  bool cv::AlignMTB::getCut()
    private static native boolean getCut_0(long nativeObj);

    // C++:  int cv::AlignMTB::getExcludeRange()
    private static native int getExcludeRange_0(long nativeObj);

    // C++:  int cv::AlignMTB::getMaxBits()
    private static native int getMaxBits_0(long nativeObj);

    // C++:  void cv::AlignMTB::computeBitmaps(Mat img, Mat& tb, Mat& eb)
    private static native void computeBitmaps_0(long nativeObj, long img_nativeObj, long tb_nativeObj, long eb_nativeObj);

    // C++:  void cv::AlignMTB::process(vector_Mat src, vector_Mat dst, Mat times, Mat response)
    private static native void process_0(long nativeObj, long src_mat_nativeObj, long dst_mat_nativeObj, long times_nativeObj, long response_nativeObj);

    // C++:  void cv::AlignMTB::process(vector_Mat src, vector_Mat dst)
    private static native void process_1(long nativeObj, long src_mat_nativeObj, long dst_mat_nativeObj);

    // C++:  void cv::AlignMTB::setCut(bool value)
    private static native void setCut_0(long nativeObj, boolean value);

    // C++:  void cv::AlignMTB::setExcludeRange(int exclude_range)
    private static native void setExcludeRange_0(long nativeObj, int exclude_range);

    // C++:  void cv::AlignMTB::setMaxBits(int max_bits)
    private static native void setMaxBits_0(long nativeObj, int max_bits);

    // C++:  void cv::AlignMTB::shiftMat(Mat src, Mat& dst, Point shift)
    private static native void shiftMat_0(long nativeObj, long src_nativeObj, long dst_nativeObj, double shift_x, double shift_y);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
