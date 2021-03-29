//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import org.opencv.features2d.Feature2D;
import org.opencv.features2d.ORB;

// C++: class ORB
/**
 * Class implementing the ORB (*oriented BRIEF*) keypoint detector and descriptor extractor
 *
 * described in CITE: RRKB11 . The algorithm uses FAST in pyramids to detect stable keypoints, selects
 * the strongest features using FAST or Harris response, finds their orientation using first-order
 * moments and computes the descriptors using BRIEF (where the coordinates of random point pairs (or
 * k-tuples) are rotated according to the measured orientation).
 */
public class ORB extends Feature2D {

    protected ORB(long addr) { super(addr); }

    // internal usage only
    public static ORB __fromPtr__(long addr) { return new ORB(addr); }

    // C++: enum ScoreType
    public static final int
            HARRIS_SCORE = 0,
            FAST_SCORE = 1;


    //
    // C++:  ORB_ScoreType cv::ORB::getScoreType()
    //

    public int getScoreType() {
        return getScoreType_0(nativeObj);
    }


    //
    // C++: static Ptr_ORB cv::ORB::create(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2, ORB_ScoreType scoreType = ORB::HARRIS_SCORE, int patchSize = 31, int fastThreshold = 20)
    //

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     @param edgeThreshold This is size of the border where the features are not detected. It should
     *     roughly match the patchSize parameter.
     *     @param firstLevel The level of pyramid to put source image to. Previous layers are filled
     *     with upscaled source image.
     *     @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     @param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller
     *     pyramid layers the perceived image area covered by a feature will be larger.
     *     @param fastThreshold the fast threshold
     * @return automatically generated
     */
    public static ORB create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize, int fastThreshold) {
        return ORB.__fromPtr__(create_0(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold));
    }

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     @param edgeThreshold This is size of the border where the features are not detected. It should
     *     roughly match the patchSize parameter.
     *     @param firstLevel The level of pyramid to put source image to. Previous layers are filled
     *     with upscaled source image.
     *     @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     @param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize) {
        return ORB.__fromPtr__(create_1(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize));
    }

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     @param edgeThreshold This is size of the border where the features are not detected. It should
     *     roughly match the patchSize parameter.
     *     @param firstLevel The level of pyramid to put source image to. Previous layers are filled
     *     with upscaled source image.
     *     @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType) {
        return ORB.__fromPtr__(create_2(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType));
    }

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     @param edgeThreshold This is size of the border where the features are not detected. It should
     *     roughly match the patchSize parameter.
     *     @param firstLevel The level of pyramid to put source image to. Previous layers are filled
     *     with upscaled source image.
     *     @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K) {
        return ORB.__fromPtr__(create_3(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K));
    }

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     @param edgeThreshold This is size of the border where the features are not detected. It should
     *     roughly match the patchSize parameter.
     *     @param firstLevel The level of pyramid to put source image to. Previous layers are filled
     *     with upscaled source image.
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel) {
        return ORB.__fromPtr__(create_4(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel));
    }

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     @param edgeThreshold This is size of the border where the features are not detected. It should
     *     roughly match the patchSize parameter.
     *     with upscaled source image.
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold) {
        return ORB.__fromPtr__(create_5(nfeatures, scaleFactor, nlevels, edgeThreshold));
    }

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     roughly match the patchSize parameter.
     *     with upscaled source image.
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create(int nfeatures, float scaleFactor, int nlevels) {
        return ORB.__fromPtr__(create_6(nfeatures, scaleFactor, nlevels));
    }

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     roughly match the patchSize parameter.
     *     with upscaled source image.
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create(int nfeatures, float scaleFactor) {
        return ORB.__fromPtr__(create_7(nfeatures, scaleFactor));
    }

    /**
     * The ORB constructor
     *
     *     @param nfeatures The maximum number of features to retain.
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     roughly match the patchSize parameter.
     *     with upscaled source image.
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create(int nfeatures) {
        return ORB.__fromPtr__(create_8(nfeatures));
    }

    /**
     * The ORB constructor
     *
     *     pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
     *     will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
     *     will mean that to cover certain scale range you will need more pyramid levels and so the speed
     *     will suffer.
     *     input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
     *     roughly match the patchSize parameter.
     *     with upscaled source image.
     *     default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
     *     so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
     *     random points (of course, those point coordinates are random, but they are generated from the
     *     pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
     *     rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
     *     output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
     *     denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
     *     bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
     *     (the score is written to KeyPoint::score and is used to retain best nfeatures features);
     *     FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
     *     but it is a little faster to compute.
     *     pyramid layers the perceived image area covered by a feature will be larger.
     * @return automatically generated
     */
    public static ORB create() {
        return ORB.__fromPtr__(create_9());
    }


    //
    // C++:  String cv::ORB::getDefaultName()
    //

    public String getDefaultName() {
        return getDefaultName_0(nativeObj);
    }


    //
    // C++:  double cv::ORB::getScaleFactor()
    //

    public double getScaleFactor() {
        return getScaleFactor_0(nativeObj);
    }


    //
    // C++:  int cv::ORB::getEdgeThreshold()
    //

    public int getEdgeThreshold() {
        return getEdgeThreshold_0(nativeObj);
    }


    //
    // C++:  int cv::ORB::getFastThreshold()
    //

    public int getFastThreshold() {
        return getFastThreshold_0(nativeObj);
    }


    //
    // C++:  int cv::ORB::getFirstLevel()
    //

    public int getFirstLevel() {
        return getFirstLevel_0(nativeObj);
    }


    //
    // C++:  int cv::ORB::getMaxFeatures()
    //

    public int getMaxFeatures() {
        return getMaxFeatures_0(nativeObj);
    }


    //
    // C++:  int cv::ORB::getNLevels()
    //

    public int getNLevels() {
        return getNLevels_0(nativeObj);
    }


    //
    // C++:  int cv::ORB::getPatchSize()
    //

    public int getPatchSize() {
        return getPatchSize_0(nativeObj);
    }


    //
    // C++:  int cv::ORB::getWTA_K()
    //

    public int getWTA_K() {
        return getWTA_K_0(nativeObj);
    }


    //
    // C++:  void cv::ORB::setEdgeThreshold(int edgeThreshold)
    //

    public void setEdgeThreshold(int edgeThreshold) {
        setEdgeThreshold_0(nativeObj, edgeThreshold);
    }


    //
    // C++:  void cv::ORB::setFastThreshold(int fastThreshold)
    //

    public void setFastThreshold(int fastThreshold) {
        setFastThreshold_0(nativeObj, fastThreshold);
    }


    //
    // C++:  void cv::ORB::setFirstLevel(int firstLevel)
    //

    public void setFirstLevel(int firstLevel) {
        setFirstLevel_0(nativeObj, firstLevel);
    }


    //
    // C++:  void cv::ORB::setMaxFeatures(int maxFeatures)
    //

    public void setMaxFeatures(int maxFeatures) {
        setMaxFeatures_0(nativeObj, maxFeatures);
    }


    //
    // C++:  void cv::ORB::setNLevels(int nlevels)
    //

    public void setNLevels(int nlevels) {
        setNLevels_0(nativeObj, nlevels);
    }


    //
    // C++:  void cv::ORB::setPatchSize(int patchSize)
    //

    public void setPatchSize(int patchSize) {
        setPatchSize_0(nativeObj, patchSize);
    }


    //
    // C++:  void cv::ORB::setScaleFactor(double scaleFactor)
    //

    public void setScaleFactor(double scaleFactor) {
        setScaleFactor_0(nativeObj, scaleFactor);
    }


    //
    // C++:  void cv::ORB::setScoreType(ORB_ScoreType scoreType)
    //

    public void setScoreType(int scoreType) {
        setScoreType_0(nativeObj, scoreType);
    }


    //
    // C++:  void cv::ORB::setWTA_K(int wta_k)
    //

    public void setWTA_K(int wta_k) {
        setWTA_K_0(nativeObj, wta_k);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  ORB_ScoreType cv::ORB::getScoreType()
    private static native int getScoreType_0(long nativeObj);

    // C++: static Ptr_ORB cv::ORB::create(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2, ORB_ScoreType scoreType = ORB::HARRIS_SCORE, int patchSize = 31, int fastThreshold = 20)
    private static native long create_0(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize, int fastThreshold);
    private static native long create_1(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize);
    private static native long create_2(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType);
    private static native long create_3(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K);
    private static native long create_4(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel);
    private static native long create_5(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold);
    private static native long create_6(int nfeatures, float scaleFactor, int nlevels);
    private static native long create_7(int nfeatures, float scaleFactor);
    private static native long create_8(int nfeatures);
    private static native long create_9();

    // C++:  String cv::ORB::getDefaultName()
    private static native String getDefaultName_0(long nativeObj);

    // C++:  double cv::ORB::getScaleFactor()
    private static native double getScaleFactor_0(long nativeObj);

    // C++:  int cv::ORB::getEdgeThreshold()
    private static native int getEdgeThreshold_0(long nativeObj);

    // C++:  int cv::ORB::getFastThreshold()
    private static native int getFastThreshold_0(long nativeObj);

    // C++:  int cv::ORB::getFirstLevel()
    private static native int getFirstLevel_0(long nativeObj);

    // C++:  int cv::ORB::getMaxFeatures()
    private static native int getMaxFeatures_0(long nativeObj);

    // C++:  int cv::ORB::getNLevels()
    private static native int getNLevels_0(long nativeObj);

    // C++:  int cv::ORB::getPatchSize()
    private static native int getPatchSize_0(long nativeObj);

    // C++:  int cv::ORB::getWTA_K()
    private static native int getWTA_K_0(long nativeObj);

    // C++:  void cv::ORB::setEdgeThreshold(int edgeThreshold)
    private static native void setEdgeThreshold_0(long nativeObj, int edgeThreshold);

    // C++:  void cv::ORB::setFastThreshold(int fastThreshold)
    private static native void setFastThreshold_0(long nativeObj, int fastThreshold);

    // C++:  void cv::ORB::setFirstLevel(int firstLevel)
    private static native void setFirstLevel_0(long nativeObj, int firstLevel);

    // C++:  void cv::ORB::setMaxFeatures(int maxFeatures)
    private static native void setMaxFeatures_0(long nativeObj, int maxFeatures);

    // C++:  void cv::ORB::setNLevels(int nlevels)
    private static native void setNLevels_0(long nativeObj, int nlevels);

    // C++:  void cv::ORB::setPatchSize(int patchSize)
    private static native void setPatchSize_0(long nativeObj, int patchSize);

    // C++:  void cv::ORB::setScaleFactor(double scaleFactor)
    private static native void setScaleFactor_0(long nativeObj, double scaleFactor);

    // C++:  void cv::ORB::setScoreType(ORB_ScoreType scoreType)
    private static native void setScoreType_0(long nativeObj, int scoreType);

    // C++:  void cv::ORB::setWTA_K(int wta_k)
    private static native void setWTA_K_0(long nativeObj, int wta_k);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
