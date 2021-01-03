//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import org.opencv.features2d.AgastFeatureDetector;
import org.opencv.features2d.Feature2D;

// C++: class AgastFeatureDetector
/**
 * Wrapping class for feature detection using the AGAST method. :
 */
public class AgastFeatureDetector extends Feature2D {

    protected AgastFeatureDetector(long addr) { super(addr); }

    // internal usage only
    public static AgastFeatureDetector __fromPtr__(long addr) { return new AgastFeatureDetector(addr); }

    // C++: enum <unnamed>
    public static final int
            THRESHOLD = 10000,
            NONMAX_SUPPRESSION = 10001;


    // C++: enum DetectorType
    public static final int
            AGAST_5_8 = 0,
            AGAST_7_12d = 1,
            AGAST_7_12s = 2,
            OAST_9_16 = 3;


    //
    // C++:  AgastFeatureDetector_DetectorType cv::AgastFeatureDetector::getType()
    //

    public int getType() {
        return getType_0(nativeObj);
    }


    //
    // C++: static Ptr_AgastFeatureDetector cv::AgastFeatureDetector::create(int threshold = 10, bool nonmaxSuppression = true, AgastFeatureDetector_DetectorType type = AgastFeatureDetector::OAST_9_16)
    //

    public static AgastFeatureDetector create(int threshold, boolean nonmaxSuppression, int type) {
        return AgastFeatureDetector.__fromPtr__(create_0(threshold, nonmaxSuppression, type));
    }

    public static AgastFeatureDetector create(int threshold, boolean nonmaxSuppression) {
        return AgastFeatureDetector.__fromPtr__(create_1(threshold, nonmaxSuppression));
    }

    public static AgastFeatureDetector create(int threshold) {
        return AgastFeatureDetector.__fromPtr__(create_2(threshold));
    }

    public static AgastFeatureDetector create() {
        return AgastFeatureDetector.__fromPtr__(create_3());
    }


    //
    // C++:  String cv::AgastFeatureDetector::getDefaultName()
    //

    public String getDefaultName() {
        return getDefaultName_0(nativeObj);
    }


    //
    // C++:  bool cv::AgastFeatureDetector::getNonmaxSuppression()
    //

    public boolean getNonmaxSuppression() {
        return getNonmaxSuppression_0(nativeObj);
    }


    //
    // C++:  int cv::AgastFeatureDetector::getThreshold()
    //

    public int getThreshold() {
        return getThreshold_0(nativeObj);
    }


    //
    // C++:  void cv::AgastFeatureDetector::setNonmaxSuppression(bool f)
    //

    public void setNonmaxSuppression(boolean f) {
        setNonmaxSuppression_0(nativeObj, f);
    }


    //
    // C++:  void cv::AgastFeatureDetector::setThreshold(int threshold)
    //

    public void setThreshold(int threshold) {
        setThreshold_0(nativeObj, threshold);
    }


    //
    // C++:  void cv::AgastFeatureDetector::setType(AgastFeatureDetector_DetectorType type)
    //

    public void setType(int type) {
        setType_0(nativeObj, type);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  AgastFeatureDetector_DetectorType cv::AgastFeatureDetector::getType()
    private static native int getType_0(long nativeObj);

    // C++: static Ptr_AgastFeatureDetector cv::AgastFeatureDetector::create(int threshold = 10, bool nonmaxSuppression = true, AgastFeatureDetector_DetectorType type = AgastFeatureDetector::OAST_9_16)
    private static native long create_0(int threshold, boolean nonmaxSuppression, int type);
    private static native long create_1(int threshold, boolean nonmaxSuppression);
    private static native long create_2(int threshold);
    private static native long create_3();

    // C++:  String cv::AgastFeatureDetector::getDefaultName()
    private static native String getDefaultName_0(long nativeObj);

    // C++:  bool cv::AgastFeatureDetector::getNonmaxSuppression()
    private static native boolean getNonmaxSuppression_0(long nativeObj);

    // C++:  int cv::AgastFeatureDetector::getThreshold()
    private static native int getThreshold_0(long nativeObj);

    // C++:  void cv::AgastFeatureDetector::setNonmaxSuppression(bool f)
    private static native void setNonmaxSuppression_0(long nativeObj, boolean f);

    // C++:  void cv::AgastFeatureDetector::setThreshold(int threshold)
    private static native void setThreshold_0(long nativeObj, int threshold);

    // C++:  void cv::AgastFeatureDetector::setType(AgastFeatureDetector_DetectorType type)
    private static native void setType_0(long nativeObj, int type);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
