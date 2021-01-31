//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import org.opencv.features2d.Feature2D;
import org.opencv.features2d.GFTTDetector;

// C++: class GFTTDetector
/**
 * Wrapping class for feature detection using the goodFeaturesToTrack function. :
 */
public class GFTTDetector extends Feature2D {

    protected GFTTDetector(long addr) { super(addr); }

    // internal usage only
    public static GFTTDetector __fromPtr__(long addr) { return new GFTTDetector(addr); }

    //
    // C++: static Ptr_GFTTDetector cv::GFTTDetector::create(int maxCorners, double qualityLevel, double minDistance, int blockSize, int gradiantSize, bool useHarrisDetector = false, double k = 0.04)
    //

    public static GFTTDetector create(int maxCorners, double qualityLevel, double minDistance, int blockSize, int gradiantSize, boolean useHarrisDetector, double k) {
        return GFTTDetector.__fromPtr__(create_0(maxCorners, qualityLevel, minDistance, blockSize, gradiantSize, useHarrisDetector, k));
    }

    public static GFTTDetector create(int maxCorners, double qualityLevel, double minDistance, int blockSize, int gradiantSize, boolean useHarrisDetector) {
        return GFTTDetector.__fromPtr__(create_1(maxCorners, qualityLevel, minDistance, blockSize, gradiantSize, useHarrisDetector));
    }

    public static GFTTDetector create(int maxCorners, double qualityLevel, double minDistance, int blockSize, int gradiantSize) {
        return GFTTDetector.__fromPtr__(create_2(maxCorners, qualityLevel, minDistance, blockSize, gradiantSize));
    }


    //
    // C++: static Ptr_GFTTDetector cv::GFTTDetector::create(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 1, int blockSize = 3, bool useHarrisDetector = false, double k = 0.04)
    //

    public static GFTTDetector create(int maxCorners, double qualityLevel, double minDistance, int blockSize, boolean useHarrisDetector, double k) {
        return GFTTDetector.__fromPtr__(create_3(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k));
    }

    public static GFTTDetector create(int maxCorners, double qualityLevel, double minDistance, int blockSize, boolean useHarrisDetector) {
        return GFTTDetector.__fromPtr__(create_4(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector));
    }

    public static GFTTDetector create(int maxCorners, double qualityLevel, double minDistance, int blockSize) {
        return GFTTDetector.__fromPtr__(create_5(maxCorners, qualityLevel, minDistance, blockSize));
    }

    public static GFTTDetector create(int maxCorners, double qualityLevel, double minDistance) {
        return GFTTDetector.__fromPtr__(create_6(maxCorners, qualityLevel, minDistance));
    }

    public static GFTTDetector create(int maxCorners, double qualityLevel) {
        return GFTTDetector.__fromPtr__(create_7(maxCorners, qualityLevel));
    }

    public static GFTTDetector create(int maxCorners) {
        return GFTTDetector.__fromPtr__(create_8(maxCorners));
    }

    public static GFTTDetector create() {
        return GFTTDetector.__fromPtr__(create_9());
    }


    //
    // C++:  String cv::GFTTDetector::getDefaultName()
    //

    public String getDefaultName() {
        return getDefaultName_0(nativeObj);
    }


    //
    // C++:  bool cv::GFTTDetector::getHarrisDetector()
    //

    public boolean getHarrisDetector() {
        return getHarrisDetector_0(nativeObj);
    }


    //
    // C++:  double cv::GFTTDetector::getK()
    //

    public double getK() {
        return getK_0(nativeObj);
    }


    //
    // C++:  double cv::GFTTDetector::getMinDistance()
    //

    public double getMinDistance() {
        return getMinDistance_0(nativeObj);
    }


    //
    // C++:  double cv::GFTTDetector::getQualityLevel()
    //

    public double getQualityLevel() {
        return getQualityLevel_0(nativeObj);
    }


    //
    // C++:  int cv::GFTTDetector::getBlockSize()
    //

    public int getBlockSize() {
        return getBlockSize_0(nativeObj);
    }


    //
    // C++:  int cv::GFTTDetector::getMaxFeatures()
    //

    public int getMaxFeatures() {
        return getMaxFeatures_0(nativeObj);
    }


    //
    // C++:  void cv::GFTTDetector::setBlockSize(int blockSize)
    //

    public void setBlockSize(int blockSize) {
        setBlockSize_0(nativeObj, blockSize);
    }


    //
    // C++:  void cv::GFTTDetector::setHarrisDetector(bool val)
    //

    public void setHarrisDetector(boolean val) {
        setHarrisDetector_0(nativeObj, val);
    }


    //
    // C++:  void cv::GFTTDetector::setK(double k)
    //

    public void setK(double k) {
        setK_0(nativeObj, k);
    }


    //
    // C++:  void cv::GFTTDetector::setMaxFeatures(int maxFeatures)
    //

    public void setMaxFeatures(int maxFeatures) {
        setMaxFeatures_0(nativeObj, maxFeatures);
    }


    //
    // C++:  void cv::GFTTDetector::setMinDistance(double minDistance)
    //

    public void setMinDistance(double minDistance) {
        setMinDistance_0(nativeObj, minDistance);
    }


    //
    // C++:  void cv::GFTTDetector::setQualityLevel(double qlevel)
    //

    public void setQualityLevel(double qlevel) {
        setQualityLevel_0(nativeObj, qlevel);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_GFTTDetector cv::GFTTDetector::create(int maxCorners, double qualityLevel, double minDistance, int blockSize, int gradiantSize, bool useHarrisDetector = false, double k = 0.04)
    private static native long create_0(int maxCorners, double qualityLevel, double minDistance, int blockSize, int gradiantSize, boolean useHarrisDetector, double k);
    private static native long create_1(int maxCorners, double qualityLevel, double minDistance, int blockSize, int gradiantSize, boolean useHarrisDetector);
    private static native long create_2(int maxCorners, double qualityLevel, double minDistance, int blockSize, int gradiantSize);

    // C++: static Ptr_GFTTDetector cv::GFTTDetector::create(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 1, int blockSize = 3, bool useHarrisDetector = false, double k = 0.04)
    private static native long create_3(int maxCorners, double qualityLevel, double minDistance, int blockSize, boolean useHarrisDetector, double k);
    private static native long create_4(int maxCorners, double qualityLevel, double minDistance, int blockSize, boolean useHarrisDetector);
    private static native long create_5(int maxCorners, double qualityLevel, double minDistance, int blockSize);
    private static native long create_6(int maxCorners, double qualityLevel, double minDistance);
    private static native long create_7(int maxCorners, double qualityLevel);
    private static native long create_8(int maxCorners);
    private static native long create_9();

    // C++:  String cv::GFTTDetector::getDefaultName()
    private static native String getDefaultName_0(long nativeObj);

    // C++:  bool cv::GFTTDetector::getHarrisDetector()
    private static native boolean getHarrisDetector_0(long nativeObj);

    // C++:  double cv::GFTTDetector::getK()
    private static native double getK_0(long nativeObj);

    // C++:  double cv::GFTTDetector::getMinDistance()
    private static native double getMinDistance_0(long nativeObj);

    // C++:  double cv::GFTTDetector::getQualityLevel()
    private static native double getQualityLevel_0(long nativeObj);

    // C++:  int cv::GFTTDetector::getBlockSize()
    private static native int getBlockSize_0(long nativeObj);

    // C++:  int cv::GFTTDetector::getMaxFeatures()
    private static native int getMaxFeatures_0(long nativeObj);

    // C++:  void cv::GFTTDetector::setBlockSize(int blockSize)
    private static native void setBlockSize_0(long nativeObj, int blockSize);

    // C++:  void cv::GFTTDetector::setHarrisDetector(bool val)
    private static native void setHarrisDetector_0(long nativeObj, boolean val);

    // C++:  void cv::GFTTDetector::setK(double k)
    private static native void setK_0(long nativeObj, double k);

    // C++:  void cv::GFTTDetector::setMaxFeatures(int maxFeatures)
    private static native void setMaxFeatures_0(long nativeObj, int maxFeatures);

    // C++:  void cv::GFTTDetector::setMinDistance(double minDistance)
    private static native void setMinDistance_0(long nativeObj, double minDistance);

    // C++:  void cv::GFTTDetector::setQualityLevel(double qlevel)
    private static native void setQualityLevel_0(long nativeObj, double qlevel);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
