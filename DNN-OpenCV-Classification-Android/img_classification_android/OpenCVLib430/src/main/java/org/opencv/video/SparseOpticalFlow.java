//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

// C++: class SparseOpticalFlow
/**
 * Base interface for sparse optical flow algorithms.
 */
public class SparseOpticalFlow extends Algorithm {

    protected SparseOpticalFlow(long addr) { super(addr); }

    // internal usage only
    public static SparseOpticalFlow __fromPtr__(long addr) { return new SparseOpticalFlow(addr); }

    //
    // C++:  void cv::SparseOpticalFlow::calc(Mat prevImg, Mat nextImg, Mat prevPts, Mat& nextPts, Mat& status, Mat& err = cv::Mat())
    //

    /**
     * Calculates a sparse optical flow.
     *
     *     @param prevImg First input image.
     *     @param nextImg Second input image of the same size and the same type as prevImg.
     *     @param prevPts Vector of 2D points for which the flow needs to be found.
     *     @param nextPts Output vector of 2D points containing the calculated new positions of input features in the second image.
     *     @param status Output status vector. Each element of the vector is set to 1 if the
     *                   flow for the corresponding features has been found. Otherwise, it is set to 0.
     *     @param err Optional output vector that contains error response for each point (inverse confidence).
     */
    public void calc(Mat prevImg, Mat nextImg, Mat prevPts, Mat nextPts, Mat status, Mat err) {
        calc_0(nativeObj, prevImg.nativeObj, nextImg.nativeObj, prevPts.nativeObj, nextPts.nativeObj, status.nativeObj, err.nativeObj);
    }

    /**
     * Calculates a sparse optical flow.
     *
     *     @param prevImg First input image.
     *     @param nextImg Second input image of the same size and the same type as prevImg.
     *     @param prevPts Vector of 2D points for which the flow needs to be found.
     *     @param nextPts Output vector of 2D points containing the calculated new positions of input features in the second image.
     *     @param status Output status vector. Each element of the vector is set to 1 if the
     *                   flow for the corresponding features has been found. Otherwise, it is set to 0.
     */
    public void calc(Mat prevImg, Mat nextImg, Mat prevPts, Mat nextPts, Mat status) {
        calc_1(nativeObj, prevImg.nativeObj, nextImg.nativeObj, prevPts.nativeObj, nextPts.nativeObj, status.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void cv::SparseOpticalFlow::calc(Mat prevImg, Mat nextImg, Mat prevPts, Mat& nextPts, Mat& status, Mat& err = cv::Mat())
    private static native void calc_0(long nativeObj, long prevImg_nativeObj, long nextImg_nativeObj, long prevPts_nativeObj, long nextPts_nativeObj, long status_nativeObj, long err_nativeObj);
    private static native void calc_1(long nativeObj, long prevImg_nativeObj, long nextImg_nativeObj, long prevPts_nativeObj, long nextPts_nativeObj, long status_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
