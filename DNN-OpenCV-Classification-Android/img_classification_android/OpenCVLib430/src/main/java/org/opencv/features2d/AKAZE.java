//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import org.opencv.features2d.AKAZE;
import org.opencv.features2d.Feature2D;

// C++: class AKAZE
/**
 * Class implementing the AKAZE keypoint detector and descriptor extractor, described in CITE: ANB13.
 *
 * AKAZE descriptors can only be used with KAZE or AKAZE keypoints. This class is thread-safe.
 *
 * <b>Note:</b> When you need descriptors use Feature2D::detectAndCompute, which
 * provides better performance. When using Feature2D::detect followed by
 * Feature2D::compute scale space pyramid is computed twice.
 *
 * <b>Note:</b> AKAZE implements T-API. When image is passed as UMat some parts of the algorithm
 * will use OpenCL.
 *
 * <b>Note:</b> [ANB13] Fast Explicit Diffusion for Accelerated Features in Nonlinear
 * Scale Spaces. Pablo F. Alcantarilla, Jesús Nuevo and Adrien Bartoli. In
 * British Machine Vision Conference (BMVC), Bristol, UK, September 2013.
 */
public class AKAZE extends Feature2D {

    protected AKAZE(long addr) { super(addr); }

    // internal usage only
    public static AKAZE __fromPtr__(long addr) { return new AKAZE(addr); }

    // C++: enum DescriptorType
    public static final int
            DESCRIPTOR_KAZE_UPRIGHT = 2,
            DESCRIPTOR_KAZE = 3,
            DESCRIPTOR_MLDB_UPRIGHT = 4,
            DESCRIPTOR_MLDB = 5;


    //
    // C++:  AKAZE_DescriptorType cv::AKAZE::getDescriptorType()
    //

    public int getDescriptorType() {
        return getDescriptorType_0(nativeObj);
    }


    //
    // C++:  KAZE_DiffusivityType cv::AKAZE::getDiffusivity()
    //

    public int getDiffusivity() {
        return getDiffusivity_0(nativeObj);
    }


    //
    // C++: static Ptr_AKAZE cv::AKAZE::create(AKAZE_DescriptorType descriptor_type = AKAZE::DESCRIPTOR_MLDB, int descriptor_size = 0, int descriptor_channels = 3, float threshold = 0.001f, int nOctaves = 4, int nOctaveLayers = 4, KAZE_DiffusivityType diffusivity = KAZE::DIFF_PM_G2)
    //

    /**
     * The AKAZE constructor
     *
     *     @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
     *     DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
     *     @param descriptor_size Size of the descriptor in bits. 0 -&gt; Full size
     *     @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
     *     @param threshold Detector response threshold to accept point
     *     @param nOctaves Maximum octave evolution of the image
     *     @param nOctaveLayers Default number of sublevels per scale level
     *     @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or
     *     DIFF_CHARBONNIER
     * @return automatically generated
     */
    public static AKAZE create(int descriptor_type, int descriptor_size, int descriptor_channels, float threshold, int nOctaves, int nOctaveLayers, int diffusivity) {
        return AKAZE.__fromPtr__(create_0(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity));
    }

    /**
     * The AKAZE constructor
     *
     *     @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
     *     DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
     *     @param descriptor_size Size of the descriptor in bits. 0 -&gt; Full size
     *     @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
     *     @param threshold Detector response threshold to accept point
     *     @param nOctaves Maximum octave evolution of the image
     *     @param nOctaveLayers Default number of sublevels per scale level
     *     DIFF_CHARBONNIER
     * @return automatically generated
     */
    public static AKAZE create(int descriptor_type, int descriptor_size, int descriptor_channels, float threshold, int nOctaves, int nOctaveLayers) {
        return AKAZE.__fromPtr__(create_1(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers));
    }

    /**
     * The AKAZE constructor
     *
     *     @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
     *     DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
     *     @param descriptor_size Size of the descriptor in bits. 0 -&gt; Full size
     *     @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
     *     @param threshold Detector response threshold to accept point
     *     @param nOctaves Maximum octave evolution of the image
     *     DIFF_CHARBONNIER
     * @return automatically generated
     */
    public static AKAZE create(int descriptor_type, int descriptor_size, int descriptor_channels, float threshold, int nOctaves) {
        return AKAZE.__fromPtr__(create_2(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves));
    }

    /**
     * The AKAZE constructor
     *
     *     @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
     *     DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
     *     @param descriptor_size Size of the descriptor in bits. 0 -&gt; Full size
     *     @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
     *     @param threshold Detector response threshold to accept point
     *     DIFF_CHARBONNIER
     * @return automatically generated
     */
    public static AKAZE create(int descriptor_type, int descriptor_size, int descriptor_channels, float threshold) {
        return AKAZE.__fromPtr__(create_3(descriptor_type, descriptor_size, descriptor_channels, threshold));
    }

    /**
     * The AKAZE constructor
     *
     *     @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
     *     DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
     *     @param descriptor_size Size of the descriptor in bits. 0 -&gt; Full size
     *     @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
     *     DIFF_CHARBONNIER
     * @return automatically generated
     */
    public static AKAZE create(int descriptor_type, int descriptor_size, int descriptor_channels) {
        return AKAZE.__fromPtr__(create_4(descriptor_type, descriptor_size, descriptor_channels));
    }

    /**
     * The AKAZE constructor
     *
     *     @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
     *     DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
     *     @param descriptor_size Size of the descriptor in bits. 0 -&gt; Full size
     *     DIFF_CHARBONNIER
     * @return automatically generated
     */
    public static AKAZE create(int descriptor_type, int descriptor_size) {
        return AKAZE.__fromPtr__(create_5(descriptor_type, descriptor_size));
    }

    /**
     * The AKAZE constructor
     *
     *     @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
     *     DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
     *     DIFF_CHARBONNIER
     * @return automatically generated
     */
    public static AKAZE create(int descriptor_type) {
        return AKAZE.__fromPtr__(create_6(descriptor_type));
    }

    /**
     * The AKAZE constructor
     *
     *     DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
     *     DIFF_CHARBONNIER
     * @return automatically generated
     */
    public static AKAZE create() {
        return AKAZE.__fromPtr__(create_7());
    }


    //
    // C++:  String cv::AKAZE::getDefaultName()
    //

    public String getDefaultName() {
        return getDefaultName_0(nativeObj);
    }


    //
    // C++:  double cv::AKAZE::getThreshold()
    //

    public double getThreshold() {
        return getThreshold_0(nativeObj);
    }


    //
    // C++:  int cv::AKAZE::getDescriptorChannels()
    //

    public int getDescriptorChannels() {
        return getDescriptorChannels_0(nativeObj);
    }


    //
    // C++:  int cv::AKAZE::getDescriptorSize()
    //

    public int getDescriptorSize() {
        return getDescriptorSize_0(nativeObj);
    }


    //
    // C++:  int cv::AKAZE::getNOctaveLayers()
    //

    public int getNOctaveLayers() {
        return getNOctaveLayers_0(nativeObj);
    }


    //
    // C++:  int cv::AKAZE::getNOctaves()
    //

    public int getNOctaves() {
        return getNOctaves_0(nativeObj);
    }


    //
    // C++:  void cv::AKAZE::setDescriptorChannels(int dch)
    //

    public void setDescriptorChannels(int dch) {
        setDescriptorChannels_0(nativeObj, dch);
    }


    //
    // C++:  void cv::AKAZE::setDescriptorSize(int dsize)
    //

    public void setDescriptorSize(int dsize) {
        setDescriptorSize_0(nativeObj, dsize);
    }


    //
    // C++:  void cv::AKAZE::setDescriptorType(AKAZE_DescriptorType dtype)
    //

    public void setDescriptorType(int dtype) {
        setDescriptorType_0(nativeObj, dtype);
    }


    //
    // C++:  void cv::AKAZE::setDiffusivity(KAZE_DiffusivityType diff)
    //

    public void setDiffusivity(int diff) {
        setDiffusivity_0(nativeObj, diff);
    }


    //
    // C++:  void cv::AKAZE::setNOctaveLayers(int octaveLayers)
    //

    public void setNOctaveLayers(int octaveLayers) {
        setNOctaveLayers_0(nativeObj, octaveLayers);
    }


    //
    // C++:  void cv::AKAZE::setNOctaves(int octaves)
    //

    public void setNOctaves(int octaves) {
        setNOctaves_0(nativeObj, octaves);
    }


    //
    // C++:  void cv::AKAZE::setThreshold(double threshold)
    //

    public void setThreshold(double threshold) {
        setThreshold_0(nativeObj, threshold);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  AKAZE_DescriptorType cv::AKAZE::getDescriptorType()
    private static native int getDescriptorType_0(long nativeObj);

    // C++:  KAZE_DiffusivityType cv::AKAZE::getDiffusivity()
    private static native int getDiffusivity_0(long nativeObj);

    // C++: static Ptr_AKAZE cv::AKAZE::create(AKAZE_DescriptorType descriptor_type = AKAZE::DESCRIPTOR_MLDB, int descriptor_size = 0, int descriptor_channels = 3, float threshold = 0.001f, int nOctaves = 4, int nOctaveLayers = 4, KAZE_DiffusivityType diffusivity = KAZE::DIFF_PM_G2)
    private static native long create_0(int descriptor_type, int descriptor_size, int descriptor_channels, float threshold, int nOctaves, int nOctaveLayers, int diffusivity);
    private static native long create_1(int descriptor_type, int descriptor_size, int descriptor_channels, float threshold, int nOctaves, int nOctaveLayers);
    private static native long create_2(int descriptor_type, int descriptor_size, int descriptor_channels, float threshold, int nOctaves);
    private static native long create_3(int descriptor_type, int descriptor_size, int descriptor_channels, float threshold);
    private static native long create_4(int descriptor_type, int descriptor_size, int descriptor_channels);
    private static native long create_5(int descriptor_type, int descriptor_size);
    private static native long create_6(int descriptor_type);
    private static native long create_7();

    // C++:  String cv::AKAZE::getDefaultName()
    private static native String getDefaultName_0(long nativeObj);

    // C++:  double cv::AKAZE::getThreshold()
    private static native double getThreshold_0(long nativeObj);

    // C++:  int cv::AKAZE::getDescriptorChannels()
    private static native int getDescriptorChannels_0(long nativeObj);

    // C++:  int cv::AKAZE::getDescriptorSize()
    private static native int getDescriptorSize_0(long nativeObj);

    // C++:  int cv::AKAZE::getNOctaveLayers()
    private static native int getNOctaveLayers_0(long nativeObj);

    // C++:  int cv::AKAZE::getNOctaves()
    private static native int getNOctaves_0(long nativeObj);

    // C++:  void cv::AKAZE::setDescriptorChannels(int dch)
    private static native void setDescriptorChannels_0(long nativeObj, int dch);

    // C++:  void cv::AKAZE::setDescriptorSize(int dsize)
    private static native void setDescriptorSize_0(long nativeObj, int dsize);

    // C++:  void cv::AKAZE::setDescriptorType(AKAZE_DescriptorType dtype)
    private static native void setDescriptorType_0(long nativeObj, int dtype);

    // C++:  void cv::AKAZE::setDiffusivity(KAZE_DiffusivityType diff)
    private static native void setDiffusivity_0(long nativeObj, int diff);

    // C++:  void cv::AKAZE::setNOctaveLayers(int octaveLayers)
    private static native void setNOctaveLayers_0(long nativeObj, int octaveLayers);

    // C++:  void cv::AKAZE::setNOctaves(int octaves)
    private static native void setNOctaves_0(long nativeObj, int octaves);

    // C++:  void cv::AKAZE::setThreshold(double threshold)
    private static native void setThreshold_0(long nativeObj, double threshold);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
