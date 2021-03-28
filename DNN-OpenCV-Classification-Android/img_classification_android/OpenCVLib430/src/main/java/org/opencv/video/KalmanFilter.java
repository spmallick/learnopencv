//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.core.Mat;

// C++: class KalmanFilter
/**
 * Kalman filter class.
 *
 * The class implements a standard Kalman filter &lt;http://en.wikipedia.org/wiki/Kalman_filter&gt;,
 * CITE: Welch95 . However, you can modify transitionMatrix, controlMatrix, and measurementMatrix to get
 * an extended Kalman filter functionality.
 * <b>Note:</b> In C API when CvKalman\* kalmanFilter structure is not needed anymore, it should be released
 * with cvReleaseKalman(&amp;kalmanFilter)
 */
public class KalmanFilter {

    protected final long nativeObj;
    protected KalmanFilter(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static KalmanFilter __fromPtr__(long addr) { return new KalmanFilter(addr); }

    //
    // C++:   cv::KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams = 0, int type = CV_32F)
    //

    /**
     *
     *     @param dynamParams Dimensionality of the state.
     *     @param measureParams Dimensionality of the measurement.
     *     @param controlParams Dimensionality of the control vector.
     *     @param type Type of the created matrices that should be CV_32F or CV_64F.
     */
    public KalmanFilter(int dynamParams, int measureParams, int controlParams, int type) {
        nativeObj = KalmanFilter_0(dynamParams, measureParams, controlParams, type);
    }

    /**
     *
     *     @param dynamParams Dimensionality of the state.
     *     @param measureParams Dimensionality of the measurement.
     *     @param controlParams Dimensionality of the control vector.
     */
    public KalmanFilter(int dynamParams, int measureParams, int controlParams) {
        nativeObj = KalmanFilter_1(dynamParams, measureParams, controlParams);
    }

    /**
     *
     *     @param dynamParams Dimensionality of the state.
     *     @param measureParams Dimensionality of the measurement.
     */
    public KalmanFilter(int dynamParams, int measureParams) {
        nativeObj = KalmanFilter_2(dynamParams, measureParams);
    }


    //
    // C++:   cv::KalmanFilter::KalmanFilter()
    //

    public KalmanFilter() {
        nativeObj = KalmanFilter_3();
    }


    //
    // C++:  Mat cv::KalmanFilter::correct(Mat measurement)
    //

    /**
     * Updates the predicted state from the measurement.
     *
     *     @param measurement The measured system parameters
     * @return automatically generated
     */
    public Mat correct(Mat measurement) {
        return new Mat(correct_0(nativeObj, measurement.nativeObj));
    }


    //
    // C++:  Mat cv::KalmanFilter::predict(Mat control = Mat())
    //

    /**
     * Computes a predicted state.
     *
     *     @param control The optional input control
     * @return automatically generated
     */
    public Mat predict(Mat control) {
        return new Mat(predict_0(nativeObj, control.nativeObj));
    }

    /**
     * Computes a predicted state.
     *
     * @return automatically generated
     */
    public Mat predict() {
        return new Mat(predict_1(nativeObj));
    }


    //
    // C++: Mat KalmanFilter::statePre
    //

    public Mat get_statePre() {
        return new Mat(get_statePre_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::statePre
    //

    public void set_statePre(Mat statePre) {
        set_statePre_0(nativeObj, statePre.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::statePost
    //

    public Mat get_statePost() {
        return new Mat(get_statePost_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::statePost
    //

    public void set_statePost(Mat statePost) {
        set_statePost_0(nativeObj, statePost.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::transitionMatrix
    //

    public Mat get_transitionMatrix() {
        return new Mat(get_transitionMatrix_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::transitionMatrix
    //

    public void set_transitionMatrix(Mat transitionMatrix) {
        set_transitionMatrix_0(nativeObj, transitionMatrix.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::controlMatrix
    //

    public Mat get_controlMatrix() {
        return new Mat(get_controlMatrix_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::controlMatrix
    //

    public void set_controlMatrix(Mat controlMatrix) {
        set_controlMatrix_0(nativeObj, controlMatrix.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::measurementMatrix
    //

    public Mat get_measurementMatrix() {
        return new Mat(get_measurementMatrix_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::measurementMatrix
    //

    public void set_measurementMatrix(Mat measurementMatrix) {
        set_measurementMatrix_0(nativeObj, measurementMatrix.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::processNoiseCov
    //

    public Mat get_processNoiseCov() {
        return new Mat(get_processNoiseCov_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::processNoiseCov
    //

    public void set_processNoiseCov(Mat processNoiseCov) {
        set_processNoiseCov_0(nativeObj, processNoiseCov.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::measurementNoiseCov
    //

    public Mat get_measurementNoiseCov() {
        return new Mat(get_measurementNoiseCov_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::measurementNoiseCov
    //

    public void set_measurementNoiseCov(Mat measurementNoiseCov) {
        set_measurementNoiseCov_0(nativeObj, measurementNoiseCov.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::errorCovPre
    //

    public Mat get_errorCovPre() {
        return new Mat(get_errorCovPre_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::errorCovPre
    //

    public void set_errorCovPre(Mat errorCovPre) {
        set_errorCovPre_0(nativeObj, errorCovPre.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::gain
    //

    public Mat get_gain() {
        return new Mat(get_gain_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::gain
    //

    public void set_gain(Mat gain) {
        set_gain_0(nativeObj, gain.nativeObj);
    }


    //
    // C++: Mat KalmanFilter::errorCovPost
    //

    public Mat get_errorCovPost() {
        return new Mat(get_errorCovPost_0(nativeObj));
    }


    //
    // C++: void KalmanFilter::errorCovPost
    //

    public void set_errorCovPost(Mat errorCovPost) {
        set_errorCovPost_0(nativeObj, errorCovPost.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams = 0, int type = CV_32F)
    private static native long KalmanFilter_0(int dynamParams, int measureParams, int controlParams, int type);
    private static native long KalmanFilter_1(int dynamParams, int measureParams, int controlParams);
    private static native long KalmanFilter_2(int dynamParams, int measureParams);

    // C++:   cv::KalmanFilter::KalmanFilter()
    private static native long KalmanFilter_3();

    // C++:  Mat cv::KalmanFilter::correct(Mat measurement)
    private static native long correct_0(long nativeObj, long measurement_nativeObj);

    // C++:  Mat cv::KalmanFilter::predict(Mat control = Mat())
    private static native long predict_0(long nativeObj, long control_nativeObj);
    private static native long predict_1(long nativeObj);

    // C++: Mat KalmanFilter::statePre
    private static native long get_statePre_0(long nativeObj);

    // C++: void KalmanFilter::statePre
    private static native void set_statePre_0(long nativeObj, long statePre_nativeObj);

    // C++: Mat KalmanFilter::statePost
    private static native long get_statePost_0(long nativeObj);

    // C++: void KalmanFilter::statePost
    private static native void set_statePost_0(long nativeObj, long statePost_nativeObj);

    // C++: Mat KalmanFilter::transitionMatrix
    private static native long get_transitionMatrix_0(long nativeObj);

    // C++: void KalmanFilter::transitionMatrix
    private static native void set_transitionMatrix_0(long nativeObj, long transitionMatrix_nativeObj);

    // C++: Mat KalmanFilter::controlMatrix
    private static native long get_controlMatrix_0(long nativeObj);

    // C++: void KalmanFilter::controlMatrix
    private static native void set_controlMatrix_0(long nativeObj, long controlMatrix_nativeObj);

    // C++: Mat KalmanFilter::measurementMatrix
    private static native long get_measurementMatrix_0(long nativeObj);

    // C++: void KalmanFilter::measurementMatrix
    private static native void set_measurementMatrix_0(long nativeObj, long measurementMatrix_nativeObj);

    // C++: Mat KalmanFilter::processNoiseCov
    private static native long get_processNoiseCov_0(long nativeObj);

    // C++: void KalmanFilter::processNoiseCov
    private static native void set_processNoiseCov_0(long nativeObj, long processNoiseCov_nativeObj);

    // C++: Mat KalmanFilter::measurementNoiseCov
    private static native long get_measurementNoiseCov_0(long nativeObj);

    // C++: void KalmanFilter::measurementNoiseCov
    private static native void set_measurementNoiseCov_0(long nativeObj, long measurementNoiseCov_nativeObj);

    // C++: Mat KalmanFilter::errorCovPre
    private static native long get_errorCovPre_0(long nativeObj);

    // C++: void KalmanFilter::errorCovPre
    private static native void set_errorCovPre_0(long nativeObj, long errorCovPre_nativeObj);

    // C++: Mat KalmanFilter::gain
    private static native long get_gain_0(long nativeObj);

    // C++: void KalmanFilter::gain
    private static native void set_gain_0(long nativeObj, long gain_nativeObj);

    // C++: Mat KalmanFilter::errorCovPost
    private static native long get_errorCovPost_0(long nativeObj);

    // C++: void KalmanFilter::errorCovPost
    private static native void set_errorCovPost_0(long nativeObj, long errorCovPost_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
