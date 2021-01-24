//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.objdetect;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

// C++: class QRCodeDetector
/**
 * Groups the object candidate rectangles.
 *     rectList  Input/output vector of rectangles. Output vector includes retained and grouped rectangles. (The Python list is not modified in place.)
 *     weights Input/output vector of weights of rectangles. Output vector includes weights of retained and grouped rectangles. (The Python list is not modified in place.)
 *     groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
 *     eps Relative difference between sides of the rectangles to merge them into a group.
 */
public class QRCodeDetector {

    protected final long nativeObj;
    protected QRCodeDetector(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static QRCodeDetector __fromPtr__(long addr) { return new QRCodeDetector(addr); }

    //
    // C++:   cv::QRCodeDetector::QRCodeDetector()
    //

    public QRCodeDetector() {
        nativeObj = QRCodeDetector_0();
    }


    //
    // C++:  bool cv::QRCodeDetector::decodeMulti(Mat img, Mat points, vector_string& decoded_info, vector_Mat& straight_qrcode = vector_Mat())
    //

    /**
     * Decodes QR codes in image once it's found by the detect() method.
     *      @param img grayscale or color (BGR) image containing QR codes.
     *      @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     *      @param points vector of Quadrangle vertices found by detect() method (or some other algorithm).
     *      @param straight_qrcode The optional output vector of images containing rectified and binarized QR codes
     * @return automatically generated
     */
    public boolean decodeMulti(Mat img, Mat points, List<String> decoded_info, List<Mat> straight_qrcode) {
        Mat straight_qrcode_mat = new Mat();
        boolean retVal = decodeMulti_0(nativeObj, img.nativeObj, points.nativeObj, decoded_info, straight_qrcode_mat.nativeObj);
        Converters.Mat_to_vector_Mat(straight_qrcode_mat, straight_qrcode);
        straight_qrcode_mat.release();
        return retVal;
    }

    /**
     * Decodes QR codes in image once it's found by the detect() method.
     *      @param img grayscale or color (BGR) image containing QR codes.
     *      @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     *      @param points vector of Quadrangle vertices found by detect() method (or some other algorithm).
     * @return automatically generated
     */
    public boolean decodeMulti(Mat img, Mat points, List<String> decoded_info) {
        return decodeMulti_1(nativeObj, img.nativeObj, points.nativeObj, decoded_info);
    }


    //
    // C++:  bool cv::QRCodeDetector::detect(Mat img, Mat& points)
    //

    /**
     * Detects QR code in image and returns the quadrangle containing the code.
     *      @param img grayscale or color (BGR) image containing (or not) QR code.
     *      @param points Output vector of vertices of the minimum-area quadrangle containing the code.
     * @return automatically generated
     */
    public boolean detect(Mat img, Mat points) {
        return detect_0(nativeObj, img.nativeObj, points.nativeObj);
    }


    //
    // C++:  bool cv::QRCodeDetector::detectAndDecodeMulti(Mat img, vector_string& decoded_info, Mat& points = Mat(), vector_Mat& straight_qrcode = vector_Mat())
    //

    /**
     * Both detects and decodes QR codes
     *     @param img grayscale or color (BGR) image containing QR codes.
     *     @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     *     @param points optional output vector of vertices of the found QR code quadrangles. Will be empty if not found.
     *     @param straight_qrcode The optional output vector of images containing rectified and binarized QR codes
     * @return automatically generated
     */
    public boolean detectAndDecodeMulti(Mat img, List<String> decoded_info, Mat points, List<Mat> straight_qrcode) {
        Mat straight_qrcode_mat = new Mat();
        boolean retVal = detectAndDecodeMulti_0(nativeObj, img.nativeObj, decoded_info, points.nativeObj, straight_qrcode_mat.nativeObj);
        Converters.Mat_to_vector_Mat(straight_qrcode_mat, straight_qrcode);
        straight_qrcode_mat.release();
        return retVal;
    }

    /**
     * Both detects and decodes QR codes
     *     @param img grayscale or color (BGR) image containing QR codes.
     *     @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     *     @param points optional output vector of vertices of the found QR code quadrangles. Will be empty if not found.
     * @return automatically generated
     */
    public boolean detectAndDecodeMulti(Mat img, List<String> decoded_info, Mat points) {
        return detectAndDecodeMulti_1(nativeObj, img.nativeObj, decoded_info, points.nativeObj);
    }

    /**
     * Both detects and decodes QR codes
     *     @param img grayscale or color (BGR) image containing QR codes.
     *     @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     * @return automatically generated
     */
    public boolean detectAndDecodeMulti(Mat img, List<String> decoded_info) {
        return detectAndDecodeMulti_2(nativeObj, img.nativeObj, decoded_info);
    }


    //
    // C++:  bool cv::QRCodeDetector::detectMulti(Mat img, Mat& points)
    //

    /**
     * Detects QR codes in image and returns the vector of the quadrangles containing the codes.
     *      @param img grayscale or color (BGR) image containing (or not) QR codes.
     *      @param points Output vector of vector of vertices of the minimum-area quadrangle containing the codes.
     * @return automatically generated
     */
    public boolean detectMulti(Mat img, Mat points) {
        return detectMulti_0(nativeObj, img.nativeObj, points.nativeObj);
    }


    //
    // C++:  string cv::QRCodeDetector::decode(Mat img, Mat points, Mat& straight_qrcode = Mat())
    //

    /**
     * Decodes QR code in image once it's found by the detect() method.
     *
     *      Returns UTF8-encoded output string or empty string if the code cannot be decoded.
     *      @param img grayscale or color (BGR) image containing QR code.
     *      @param points Quadrangle vertices found by detect() method (or some other algorithm).
     *      @param straight_qrcode The optional output image containing rectified and binarized QR code
     * @return automatically generated
     */
    public String decode(Mat img, Mat points, Mat straight_qrcode) {
        return decode_0(nativeObj, img.nativeObj, points.nativeObj, straight_qrcode.nativeObj);
    }

    /**
     * Decodes QR code in image once it's found by the detect() method.
     *
     *      Returns UTF8-encoded output string or empty string if the code cannot be decoded.
     *      @param img grayscale or color (BGR) image containing QR code.
     *      @param points Quadrangle vertices found by detect() method (or some other algorithm).
     * @return automatically generated
     */
    public String decode(Mat img, Mat points) {
        return decode_1(nativeObj, img.nativeObj, points.nativeObj);
    }


    //
    // C++:  string cv::QRCodeDetector::detectAndDecode(Mat img, Mat& points = Mat(), Mat& straight_qrcode = Mat())
    //

    /**
     * Both detects and decodes QR code
     *
     *      @param img grayscale or color (BGR) image containing QR code.
     *      @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
     *      @param straight_qrcode The optional output image containing rectified and binarized QR code
     * @return automatically generated
     */
    public String detectAndDecode(Mat img, Mat points, Mat straight_qrcode) {
        return detectAndDecode_0(nativeObj, img.nativeObj, points.nativeObj, straight_qrcode.nativeObj);
    }

    /**
     * Both detects and decodes QR code
     *
     *      @param img grayscale or color (BGR) image containing QR code.
     *      @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
     * @return automatically generated
     */
    public String detectAndDecode(Mat img, Mat points) {
        return detectAndDecode_1(nativeObj, img.nativeObj, points.nativeObj);
    }

    /**
     * Both detects and decodes QR code
     *
     *      @param img grayscale or color (BGR) image containing QR code.
     * @return automatically generated
     */
    public String detectAndDecode(Mat img) {
        return detectAndDecode_2(nativeObj, img.nativeObj);
    }


    //
    // C++:  void cv::QRCodeDetector::setEpsX(double epsX)
    //

    /**
     * sets the epsilon used during the horizontal scan of QR code stop marker detection.
     *      @param epsX Epsilon neighborhood, which allows you to determine the horizontal pattern
     *      of the scheme 1:1:3:1:1 according to QR code standard.
     */
    public void setEpsX(double epsX) {
        setEpsX_0(nativeObj, epsX);
    }


    //
    // C++:  void cv::QRCodeDetector::setEpsY(double epsY)
    //

    /**
     * sets the epsilon used during the vertical scan of QR code stop marker detection.
     *      @param epsY Epsilon neighborhood, which allows you to determine the vertical pattern
     *      of the scheme 1:1:3:1:1 according to QR code standard.
     */
    public void setEpsY(double epsY) {
        setEpsY_0(nativeObj, epsY);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::QRCodeDetector::QRCodeDetector()
    private static native long QRCodeDetector_0();

    // C++:  bool cv::QRCodeDetector::decodeMulti(Mat img, Mat points, vector_string& decoded_info, vector_Mat& straight_qrcode = vector_Mat())
    private static native boolean decodeMulti_0(long nativeObj, long img_nativeObj, long points_nativeObj, List<String> decoded_info, long straight_qrcode_mat_nativeObj);
    private static native boolean decodeMulti_1(long nativeObj, long img_nativeObj, long points_nativeObj, List<String> decoded_info);

    // C++:  bool cv::QRCodeDetector::detect(Mat img, Mat& points)
    private static native boolean detect_0(long nativeObj, long img_nativeObj, long points_nativeObj);

    // C++:  bool cv::QRCodeDetector::detectAndDecodeMulti(Mat img, vector_string& decoded_info, Mat& points = Mat(), vector_Mat& straight_qrcode = vector_Mat())
    private static native boolean detectAndDecodeMulti_0(long nativeObj, long img_nativeObj, List<String> decoded_info, long points_nativeObj, long straight_qrcode_mat_nativeObj);
    private static native boolean detectAndDecodeMulti_1(long nativeObj, long img_nativeObj, List<String> decoded_info, long points_nativeObj);
    private static native boolean detectAndDecodeMulti_2(long nativeObj, long img_nativeObj, List<String> decoded_info);

    // C++:  bool cv::QRCodeDetector::detectMulti(Mat img, Mat& points)
    private static native boolean detectMulti_0(long nativeObj, long img_nativeObj, long points_nativeObj);

    // C++:  string cv::QRCodeDetector::decode(Mat img, Mat points, Mat& straight_qrcode = Mat())
    private static native String decode_0(long nativeObj, long img_nativeObj, long points_nativeObj, long straight_qrcode_nativeObj);
    private static native String decode_1(long nativeObj, long img_nativeObj, long points_nativeObj);

    // C++:  string cv::QRCodeDetector::detectAndDecode(Mat img, Mat& points = Mat(), Mat& straight_qrcode = Mat())
    private static native String detectAndDecode_0(long nativeObj, long img_nativeObj, long points_nativeObj, long straight_qrcode_nativeObj);
    private static native String detectAndDecode_1(long nativeObj, long img_nativeObj, long points_nativeObj);
    private static native String detectAndDecode_2(long nativeObj, long img_nativeObj);

    // C++:  void cv::QRCodeDetector::setEpsX(double epsX)
    private static native void setEpsX_0(long nativeObj, double epsX);

    // C++:  void cv::QRCodeDetector::setEpsY(double epsY)
    private static native void setEpsY_0(long nativeObj, double epsY);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
