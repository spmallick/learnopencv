//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.video.SparseOpticalFlow;
import org.opencv.video.SparsePyrLKOpticalFlow;

// C++: class SparsePyrLKOpticalFlow
/**
 * Class used for calculating a sparse optical flow.
 *
 * The class can calculate an optical flow for a sparse feature set using the
 * iterative Lucas-Kanade method with pyramids.
 *
 * SEE: calcOpticalFlowPyrLK
 */
public class SparsePyrLKOpticalFlow extends SparseOpticalFlow {

    protected SparsePyrLKOpticalFlow(long addr) { super(addr); }

    // internal usage only
    public static SparsePyrLKOpticalFlow __fromPtr__(long addr) { return new SparsePyrLKOpticalFlow(addr); }

    //
    // C++: static Ptr_SparsePyrLKOpticalFlow cv::SparsePyrLKOpticalFlow::create(Size winSize = Size(21, 21), int maxLevel = 3, TermCriteria crit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags = 0, double minEigThreshold = 1e-4)
    //

    public static SparsePyrLKOpticalFlow create(Size winSize, int maxLevel, TermCriteria crit, int flags, double minEigThreshold) {
        return SparsePyrLKOpticalFlow.__fromPtr__(create_0(winSize.width, winSize.height, maxLevel, crit.type, crit.maxCount, crit.epsilon, flags, minEigThreshold));
    }

    public static SparsePyrLKOpticalFlow create(Size winSize, int maxLevel, TermCriteria crit, int flags) {
        return SparsePyrLKOpticalFlow.__fromPtr__(create_1(winSize.width, winSize.height, maxLevel, crit.type, crit.maxCount, crit.epsilon, flags));
    }

    public static SparsePyrLKOpticalFlow create(Size winSize, int maxLevel, TermCriteria crit) {
        return SparsePyrLKOpticalFlow.__fromPtr__(create_2(winSize.width, winSize.height, maxLevel, crit.type, crit.maxCount, crit.epsilon));
    }

    public static SparsePyrLKOpticalFlow create(Size winSize, int maxLevel) {
        return SparsePyrLKOpticalFlow.__fromPtr__(create_3(winSize.width, winSize.height, maxLevel));
    }

    public static SparsePyrLKOpticalFlow create(Size winSize) {
        return SparsePyrLKOpticalFlow.__fromPtr__(create_4(winSize.width, winSize.height));
    }

    public static SparsePyrLKOpticalFlow create() {
        return SparsePyrLKOpticalFlow.__fromPtr__(create_5());
    }


    //
    // C++:  Size cv::SparsePyrLKOpticalFlow::getWinSize()
    //

    public Size getWinSize() {
        return new Size(getWinSize_0(nativeObj));
    }


    //
    // C++:  TermCriteria cv::SparsePyrLKOpticalFlow::getTermCriteria()
    //

    public TermCriteria getTermCriteria() {
        return new TermCriteria(getTermCriteria_0(nativeObj));
    }


    //
    // C++:  double cv::SparsePyrLKOpticalFlow::getMinEigThreshold()
    //

    public double getMinEigThreshold() {
        return getMinEigThreshold_0(nativeObj);
    }


    //
    // C++:  int cv::SparsePyrLKOpticalFlow::getFlags()
    //

    public int getFlags() {
        return getFlags_0(nativeObj);
    }


    //
    // C++:  int cv::SparsePyrLKOpticalFlow::getMaxLevel()
    //

    public int getMaxLevel() {
        return getMaxLevel_0(nativeObj);
    }


    //
    // C++:  void cv::SparsePyrLKOpticalFlow::setFlags(int flags)
    //

    public void setFlags(int flags) {
        setFlags_0(nativeObj, flags);
    }


    //
    // C++:  void cv::SparsePyrLKOpticalFlow::setMaxLevel(int maxLevel)
    //

    public void setMaxLevel(int maxLevel) {
        setMaxLevel_0(nativeObj, maxLevel);
    }


    //
    // C++:  void cv::SparsePyrLKOpticalFlow::setMinEigThreshold(double minEigThreshold)
    //

    public void setMinEigThreshold(double minEigThreshold) {
        setMinEigThreshold_0(nativeObj, minEigThreshold);
    }


    //
    // C++:  void cv::SparsePyrLKOpticalFlow::setTermCriteria(TermCriteria crit)
    //

    public void setTermCriteria(TermCriteria crit) {
        setTermCriteria_0(nativeObj, crit.type, crit.maxCount, crit.epsilon);
    }


    //
    // C++:  void cv::SparsePyrLKOpticalFlow::setWinSize(Size winSize)
    //

    public void setWinSize(Size winSize) {
        setWinSize_0(nativeObj, winSize.width, winSize.height);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_SparsePyrLKOpticalFlow cv::SparsePyrLKOpticalFlow::create(Size winSize = Size(21, 21), int maxLevel = 3, TermCriteria crit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags = 0, double minEigThreshold = 1e-4)
    private static native long create_0(double winSize_width, double winSize_height, int maxLevel, int crit_type, int crit_maxCount, double crit_epsilon, int flags, double minEigThreshold);
    private static native long create_1(double winSize_width, double winSize_height, int maxLevel, int crit_type, int crit_maxCount, double crit_epsilon, int flags);
    private static native long create_2(double winSize_width, double winSize_height, int maxLevel, int crit_type, int crit_maxCount, double crit_epsilon);
    private static native long create_3(double winSize_width, double winSize_height, int maxLevel);
    private static native long create_4(double winSize_width, double winSize_height);
    private static native long create_5();

    // C++:  Size cv::SparsePyrLKOpticalFlow::getWinSize()
    private static native double[] getWinSize_0(long nativeObj);

    // C++:  TermCriteria cv::SparsePyrLKOpticalFlow::getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  double cv::SparsePyrLKOpticalFlow::getMinEigThreshold()
    private static native double getMinEigThreshold_0(long nativeObj);

    // C++:  int cv::SparsePyrLKOpticalFlow::getFlags()
    private static native int getFlags_0(long nativeObj);

    // C++:  int cv::SparsePyrLKOpticalFlow::getMaxLevel()
    private static native int getMaxLevel_0(long nativeObj);

    // C++:  void cv::SparsePyrLKOpticalFlow::setFlags(int flags)
    private static native void setFlags_0(long nativeObj, int flags);

    // C++:  void cv::SparsePyrLKOpticalFlow::setMaxLevel(int maxLevel)
    private static native void setMaxLevel_0(long nativeObj, int maxLevel);

    // C++:  void cv::SparsePyrLKOpticalFlow::setMinEigThreshold(double minEigThreshold)
    private static native void setMinEigThreshold_0(long nativeObj, double minEigThreshold);

    // C++:  void cv::SparsePyrLKOpticalFlow::setTermCriteria(TermCriteria crit)
    private static native void setTermCriteria_0(long nativeObj, int crit_type, int crit_maxCount, double crit_epsilon);

    // C++:  void cv::SparsePyrLKOpticalFlow::setWinSize(Size winSize)
    private static native void setWinSize_0(long nativeObj, double winSize_width, double winSize_height);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
