//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.imgproc;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.core.Size;

// C++: class LineSegmentDetector
/**
 * Line segment detector class
 *
 * following the algorithm described at CITE: Rafael12 .
 *
 * <b>Note:</b> Implementation has been removed due original code license conflict
 */
public class LineSegmentDetector extends Algorithm {

    protected LineSegmentDetector(long addr) { super(addr); }

    // internal usage only
    public static LineSegmentDetector __fromPtr__(long addr) { return new LineSegmentDetector(addr); }

    //
    // C++:  int cv::LineSegmentDetector::compareSegments(Size size, Mat lines1, Mat lines2, Mat& _image = Mat())
    //

    /**
     * Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.
     *
     *     @param size The size of the image, where lines1 and lines2 were found.
     *     @param lines1 The first group of lines that needs to be drawn. It is visualized in blue color.
     *     @param lines2 The second group of lines. They visualized in red color.
     *     @param _image Optional image, where the lines will be drawn. The image should be color(3-channel)
     *     in order for lines1 and lines2 to be drawn in the above mentioned colors.
     * @return automatically generated
     */
    public int compareSegments(Size size, Mat lines1, Mat lines2, Mat _image) {
        return compareSegments_0(nativeObj, size.width, size.height, lines1.nativeObj, lines2.nativeObj, _image.nativeObj);
    }

    /**
     * Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.
     *
     *     @param size The size of the image, where lines1 and lines2 were found.
     *     @param lines1 The first group of lines that needs to be drawn. It is visualized in blue color.
     *     @param lines2 The second group of lines. They visualized in red color.
     *     in order for lines1 and lines2 to be drawn in the above mentioned colors.
     * @return automatically generated
     */
    public int compareSegments(Size size, Mat lines1, Mat lines2) {
        return compareSegments_1(nativeObj, size.width, size.height, lines1.nativeObj, lines2.nativeObj);
    }


    //
    // C++:  void cv::LineSegmentDetector::detect(Mat _image, Mat& _lines, Mat& width = Mat(), Mat& prec = Mat(), Mat& nfa = Mat())
    //

    /**
     * Finds lines in the input image.
     *
     *     This is the output of the default parameters of the algorithm on the above shown image.
     *
     *     ![image](pics/building_lsd.png)
     *
     *     @param _image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use:
     *     {@code lsd_ptr-&gt;detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);}
     *     @param _lines A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line. Where
     *     Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly
     *     oriented depending on the gradient.
     *     @param width Vector of widths of the regions, where the lines are found. E.g. Width of line.
     *     @param prec Vector of precisions with which the lines are found.
     *     @param nfa Vector containing number of false alarms in the line region, with precision of 10%. The
     *     bigger the value, logarithmically better the detection.
     * <ul>
     *   <li>
     *      -1 corresponds to 10 mean false alarms
     *   </li>
     *   <li>
     *      0 corresponds to 1 mean false alarm
     *   </li>
     *   <li>
     *      1 corresponds to 0.1 mean false alarms
     *     This vector will be calculated only when the objects type is #LSD_REFINE_ADV.
     *   </li>
     * </ul>
     */
    public void detect(Mat _image, Mat _lines, Mat width, Mat prec, Mat nfa) {
        detect_0(nativeObj, _image.nativeObj, _lines.nativeObj, width.nativeObj, prec.nativeObj, nfa.nativeObj);
    }

    /**
     * Finds lines in the input image.
     *
     *     This is the output of the default parameters of the algorithm on the above shown image.
     *
     *     ![image](pics/building_lsd.png)
     *
     *     @param _image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use:
     *     {@code lsd_ptr-&gt;detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);}
     *     @param _lines A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line. Where
     *     Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly
     *     oriented depending on the gradient.
     *     @param width Vector of widths of the regions, where the lines are found. E.g. Width of line.
     *     @param prec Vector of precisions with which the lines are found.
     *     bigger the value, logarithmically better the detection.
     * <ul>
     *   <li>
     *      -1 corresponds to 10 mean false alarms
     *   </li>
     *   <li>
     *      0 corresponds to 1 mean false alarm
     *   </li>
     *   <li>
     *      1 corresponds to 0.1 mean false alarms
     *     This vector will be calculated only when the objects type is #LSD_REFINE_ADV.
     *   </li>
     * </ul>
     */
    public void detect(Mat _image, Mat _lines, Mat width, Mat prec) {
        detect_1(nativeObj, _image.nativeObj, _lines.nativeObj, width.nativeObj, prec.nativeObj);
    }

    /**
     * Finds lines in the input image.
     *
     *     This is the output of the default parameters of the algorithm on the above shown image.
     *
     *     ![image](pics/building_lsd.png)
     *
     *     @param _image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use:
     *     {@code lsd_ptr-&gt;detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);}
     *     @param _lines A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line. Where
     *     Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly
     *     oriented depending on the gradient.
     *     @param width Vector of widths of the regions, where the lines are found. E.g. Width of line.
     *     bigger the value, logarithmically better the detection.
     * <ul>
     *   <li>
     *      -1 corresponds to 10 mean false alarms
     *   </li>
     *   <li>
     *      0 corresponds to 1 mean false alarm
     *   </li>
     *   <li>
     *      1 corresponds to 0.1 mean false alarms
     *     This vector will be calculated only when the objects type is #LSD_REFINE_ADV.
     *   </li>
     * </ul>
     */
    public void detect(Mat _image, Mat _lines, Mat width) {
        detect_2(nativeObj, _image.nativeObj, _lines.nativeObj, width.nativeObj);
    }

    /**
     * Finds lines in the input image.
     *
     *     This is the output of the default parameters of the algorithm on the above shown image.
     *
     *     ![image](pics/building_lsd.png)
     *
     *     @param _image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use:
     *     {@code lsd_ptr-&gt;detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);}
     *     @param _lines A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line. Where
     *     Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly
     *     oriented depending on the gradient.
     *     bigger the value, logarithmically better the detection.
     * <ul>
     *   <li>
     *      -1 corresponds to 10 mean false alarms
     *   </li>
     *   <li>
     *      0 corresponds to 1 mean false alarm
     *   </li>
     *   <li>
     *      1 corresponds to 0.1 mean false alarms
     *     This vector will be calculated only when the objects type is #LSD_REFINE_ADV.
     *   </li>
     * </ul>
     */
    public void detect(Mat _image, Mat _lines) {
        detect_3(nativeObj, _image.nativeObj, _lines.nativeObj);
    }


    //
    // C++:  void cv::LineSegmentDetector::drawSegments(Mat& _image, Mat lines)
    //

    /**
     * Draws the line segments on a given image.
     *     @param _image The image, where the lines will be drawn. Should be bigger or equal to the image,
     *     where the lines were found.
     *     @param lines A vector of the lines that needed to be drawn.
     */
    public void drawSegments(Mat _image, Mat lines) {
        drawSegments_0(nativeObj, _image.nativeObj, lines.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  int cv::LineSegmentDetector::compareSegments(Size size, Mat lines1, Mat lines2, Mat& _image = Mat())
    private static native int compareSegments_0(long nativeObj, double size_width, double size_height, long lines1_nativeObj, long lines2_nativeObj, long _image_nativeObj);
    private static native int compareSegments_1(long nativeObj, double size_width, double size_height, long lines1_nativeObj, long lines2_nativeObj);

    // C++:  void cv::LineSegmentDetector::detect(Mat _image, Mat& _lines, Mat& width = Mat(), Mat& prec = Mat(), Mat& nfa = Mat())
    private static native void detect_0(long nativeObj, long _image_nativeObj, long _lines_nativeObj, long width_nativeObj, long prec_nativeObj, long nfa_nativeObj);
    private static native void detect_1(long nativeObj, long _image_nativeObj, long _lines_nativeObj, long width_nativeObj, long prec_nativeObj);
    private static native void detect_2(long nativeObj, long _image_nativeObj, long _lines_nativeObj, long width_nativeObj);
    private static native void detect_3(long nativeObj, long _image_nativeObj, long _lines_nativeObj);

    // C++:  void cv::LineSegmentDetector::drawSegments(Mat& _image, Mat lines)
    private static native void drawSegments_0(long nativeObj, long _image_nativeObj, long lines_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
