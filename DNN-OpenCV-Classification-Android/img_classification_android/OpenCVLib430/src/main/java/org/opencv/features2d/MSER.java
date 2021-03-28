//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.MSER;
import org.opencv.utils.Converters;

// C++: class MSER
/**
 * Maximally stable extremal region extractor
 *
 * The class encapsulates all the parameters of the %MSER extraction algorithm (see [wiki
 * article](http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions)).
 *
 * <ul>
 *   <li>
 *  there are two different implementation of %MSER: one for grey image, one for color image
 *   </li>
 * </ul>
 *
 * <ul>
 *   <li>
 *  the grey image algorithm is taken from: CITE: nister2008linear ;  the paper claims to be faster
 * than union-find method; it actually get 1.5~2m/s on my centrino L7200 1.2GHz laptop.
 *   </li>
 * </ul>
 *
 * <ul>
 *   <li>
 *  the color image algorithm is taken from: CITE: forssen2007maximally ; it should be much slower
 * than grey image method ( 3~4 times ); the chi_table.h file is taken directly from paper's source
 * code which is distributed under GPL.
 *   </li>
 * </ul>
 *
 * <ul>
 *   <li>
 *  (Python) A complete example showing the use of the %MSER detector can be found at samples/python/mser.py
 *   </li>
 * </ul>
 */
public class MSER extends Feature2D {

    protected MSER(long addr) { super(addr); }

    // internal usage only
    public static MSER __fromPtr__(long addr) { return new MSER(addr); }

    //
    // C++: static Ptr_MSER cv::MSER::create(int _delta = 5, int _min_area = 60, int _max_area = 14400, double _max_variation = 0.25, double _min_diversity = .2, int _max_evolution = 200, double _area_threshold = 1.01, double _min_margin = 0.003, int _edge_blur_size = 5)
    //

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     *     @param _min_area prune the area which smaller than minArea
     *     @param _max_area prune the area which bigger than maxArea
     *     @param _max_variation prune the area have similar size to its children
     *     @param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
     *     @param _max_evolution  for color image, the evolution steps
     *     @param _area_threshold for color image, the area threshold to cause re-initialize
     *     @param _min_margin for color image, ignore too small margin
     *     @param _edge_blur_size for color image, the aperture size for edge blur
     * @return automatically generated
     */
    public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin, int _edge_blur_size) {
        return MSER.__fromPtr__(create_0(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold, _min_margin, _edge_blur_size));
    }

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     *     @param _min_area prune the area which smaller than minArea
     *     @param _max_area prune the area which bigger than maxArea
     *     @param _max_variation prune the area have similar size to its children
     *     @param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
     *     @param _max_evolution  for color image, the evolution steps
     *     @param _area_threshold for color image, the area threshold to cause re-initialize
     *     @param _min_margin for color image, ignore too small margin
     * @return automatically generated
     */
    public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin) {
        return MSER.__fromPtr__(create_1(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold, _min_margin));
    }

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     *     @param _min_area prune the area which smaller than minArea
     *     @param _max_area prune the area which bigger than maxArea
     *     @param _max_variation prune the area have similar size to its children
     *     @param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
     *     @param _max_evolution  for color image, the evolution steps
     *     @param _area_threshold for color image, the area threshold to cause re-initialize
     * @return automatically generated
     */
    public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold) {
        return MSER.__fromPtr__(create_2(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold));
    }

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     *     @param _min_area prune the area which smaller than minArea
     *     @param _max_area prune the area which bigger than maxArea
     *     @param _max_variation prune the area have similar size to its children
     *     @param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
     *     @param _max_evolution  for color image, the evolution steps
     * @return automatically generated
     */
    public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution) {
        return MSER.__fromPtr__(create_3(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution));
    }

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     *     @param _min_area prune the area which smaller than minArea
     *     @param _max_area prune the area which bigger than maxArea
     *     @param _max_variation prune the area have similar size to its children
     *     @param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
     * @return automatically generated
     */
    public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity) {
        return MSER.__fromPtr__(create_4(_delta, _min_area, _max_area, _max_variation, _min_diversity));
    }

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     *     @param _min_area prune the area which smaller than minArea
     *     @param _max_area prune the area which bigger than maxArea
     *     @param _max_variation prune the area have similar size to its children
     * @return automatically generated
     */
    public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation) {
        return MSER.__fromPtr__(create_5(_delta, _min_area, _max_area, _max_variation));
    }

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     *     @param _min_area prune the area which smaller than minArea
     *     @param _max_area prune the area which bigger than maxArea
     * @return automatically generated
     */
    public static MSER create(int _delta, int _min_area, int _max_area) {
        return MSER.__fromPtr__(create_6(_delta, _min_area, _max_area));
    }

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     *     @param _min_area prune the area which smaller than minArea
     * @return automatically generated
     */
    public static MSER create(int _delta, int _min_area) {
        return MSER.__fromPtr__(create_7(_delta, _min_area));
    }

    /**
     * Full constructor for %MSER detector
     *
     *     @param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
     * @return automatically generated
     */
    public static MSER create(int _delta) {
        return MSER.__fromPtr__(create_8(_delta));
    }

    /**
     * Full constructor for %MSER detector
     *
     * @return automatically generated
     */
    public static MSER create() {
        return MSER.__fromPtr__(create_9());
    }


    //
    // C++:  String cv::MSER::getDefaultName()
    //

    public String getDefaultName() {
        return getDefaultName_0(nativeObj);
    }


    //
    // C++:  bool cv::MSER::getPass2Only()
    //

    public boolean getPass2Only() {
        return getPass2Only_0(nativeObj);
    }


    //
    // C++:  int cv::MSER::getDelta()
    //

    public int getDelta() {
        return getDelta_0(nativeObj);
    }


    //
    // C++:  int cv::MSER::getMaxArea()
    //

    public int getMaxArea() {
        return getMaxArea_0(nativeObj);
    }


    //
    // C++:  int cv::MSER::getMinArea()
    //

    public int getMinArea() {
        return getMinArea_0(nativeObj);
    }


    //
    // C++:  void cv::MSER::detectRegions(Mat image, vector_vector_Point& msers, vector_Rect& bboxes)
    //

    /**
     * Detect %MSER regions
     *
     *     @param image input image (8UC1, 8UC3 or 8UC4, must be greater or equal than 3x3)
     *     @param msers resulting list of point sets
     *     @param bboxes resulting bounding boxes
     */
    public void detectRegions(Mat image, List<MatOfPoint> msers, MatOfRect bboxes) {
        Mat msers_mat = new Mat();
        Mat bboxes_mat = bboxes;
        detectRegions_0(nativeObj, image.nativeObj, msers_mat.nativeObj, bboxes_mat.nativeObj);
        Converters.Mat_to_vector_vector_Point(msers_mat, msers);
        msers_mat.release();
    }


    //
    // C++:  void cv::MSER::setDelta(int delta)
    //

    public void setDelta(int delta) {
        setDelta_0(nativeObj, delta);
    }


    //
    // C++:  void cv::MSER::setMaxArea(int maxArea)
    //

    public void setMaxArea(int maxArea) {
        setMaxArea_0(nativeObj, maxArea);
    }


    //
    // C++:  void cv::MSER::setMinArea(int minArea)
    //

    public void setMinArea(int minArea) {
        setMinArea_0(nativeObj, minArea);
    }


    //
    // C++:  void cv::MSER::setPass2Only(bool f)
    //

    public void setPass2Only(boolean f) {
        setPass2Only_0(nativeObj, f);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_MSER cv::MSER::create(int _delta = 5, int _min_area = 60, int _max_area = 14400, double _max_variation = 0.25, double _min_diversity = .2, int _max_evolution = 200, double _area_threshold = 1.01, double _min_margin = 0.003, int _edge_blur_size = 5)
    private static native long create_0(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin, int _edge_blur_size);
    private static native long create_1(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin);
    private static native long create_2(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold);
    private static native long create_3(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution);
    private static native long create_4(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity);
    private static native long create_5(int _delta, int _min_area, int _max_area, double _max_variation);
    private static native long create_6(int _delta, int _min_area, int _max_area);
    private static native long create_7(int _delta, int _min_area);
    private static native long create_8(int _delta);
    private static native long create_9();

    // C++:  String cv::MSER::getDefaultName()
    private static native String getDefaultName_0(long nativeObj);

    // C++:  bool cv::MSER::getPass2Only()
    private static native boolean getPass2Only_0(long nativeObj);

    // C++:  int cv::MSER::getDelta()
    private static native int getDelta_0(long nativeObj);

    // C++:  int cv::MSER::getMaxArea()
    private static native int getMaxArea_0(long nativeObj);

    // C++:  int cv::MSER::getMinArea()
    private static native int getMinArea_0(long nativeObj);

    // C++:  void cv::MSER::detectRegions(Mat image, vector_vector_Point& msers, vector_Rect& bboxes)
    private static native void detectRegions_0(long nativeObj, long image_nativeObj, long msers_mat_nativeObj, long bboxes_mat_nativeObj);

    // C++:  void cv::MSER::setDelta(int delta)
    private static native void setDelta_0(long nativeObj, int delta);

    // C++:  void cv::MSER::setMaxArea(int maxArea)
    private static native void setMaxArea_0(long nativeObj, int maxArea);

    // C++:  void cv::MSER::setMinArea(int minArea)
    private static native void setMinArea_0(long nativeObj, int minArea);

    // C++:  void cv::MSER::setPass2Only(bool f)
    private static native void setPass2Only_0(long nativeObj, boolean f);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
