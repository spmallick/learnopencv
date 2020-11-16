//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.imgproc;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.GeneralizedHoughBallard;
import org.opencv.imgproc.GeneralizedHoughGuil;
import org.opencv.imgproc.LineSegmentDetector;
import org.opencv.utils.Converters;

// C++: class Imgproc

public class Imgproc {

    private static final int
            IPL_BORDER_CONSTANT = 0,
            IPL_BORDER_REPLICATE = 1,
            IPL_BORDER_REFLECT = 2,
            IPL_BORDER_WRAP = 3,
            IPL_BORDER_REFLECT_101 = 4,
            IPL_BORDER_TRANSPARENT = 5,
            CV_INTER_NN = 0,
            CV_INTER_LINEAR = 1,
            CV_INTER_CUBIC = 2,
            CV_INTER_AREA = 3,
            CV_INTER_LANCZOS4 = 4,
            CV_MOP_ERODE = 0,
            CV_MOP_DILATE = 1,
            CV_MOP_OPEN = 2,
            CV_MOP_CLOSE = 3,
            CV_MOP_GRADIENT = 4,
            CV_MOP_TOPHAT = 5,
            CV_MOP_BLACKHAT = 6,
            CV_RETR_EXTERNAL = 0,
            CV_RETR_LIST = 1,
            CV_RETR_CCOMP = 2,
            CV_RETR_TREE = 3,
            CV_RETR_FLOODFILL = 4,
            CV_CHAIN_APPROX_NONE = 1,
            CV_CHAIN_APPROX_SIMPLE = 2,
            CV_CHAIN_APPROX_TC89_L1 = 3,
            CV_CHAIN_APPROX_TC89_KCOS = 4,
            CV_THRESH_BINARY = 0,
            CV_THRESH_BINARY_INV = 1,
            CV_THRESH_TRUNC = 2,
            CV_THRESH_TOZERO = 3,
            CV_THRESH_TOZERO_INV = 4,
            CV_THRESH_MASK = 7,
            CV_THRESH_OTSU = 8,
            CV_THRESH_TRIANGLE = 16;


    // C++: enum HersheyFonts
    public static final int
            FONT_HERSHEY_SIMPLEX = 0,
            FONT_HERSHEY_PLAIN = 1,
            FONT_HERSHEY_DUPLEX = 2,
            FONT_HERSHEY_COMPLEX = 3,
            FONT_HERSHEY_TRIPLEX = 4,
            FONT_HERSHEY_COMPLEX_SMALL = 5,
            FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
            FONT_HERSHEY_SCRIPT_COMPLEX = 7,
            FONT_ITALIC = 16;


    // C++: enum InterpolationMasks
    public static final int
            INTER_BITS = 5,
            INTER_BITS2 = INTER_BITS * 2,
            INTER_TAB_SIZE = 1 << INTER_BITS,
            INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE;


    // C++: enum MorphTypes
    public static final int
            MORPH_ERODE = 0,
            MORPH_DILATE = 1,
            MORPH_OPEN = 2,
            MORPH_CLOSE = 3,
            MORPH_GRADIENT = 4,
            MORPH_TOPHAT = 5,
            MORPH_BLACKHAT = 6,
            MORPH_HITMISS = 7;


    // C++: enum FloodFillFlags
    public static final int
            FLOODFILL_FIXED_RANGE = 1 << 16,
            FLOODFILL_MASK_ONLY = 1 << 17;


    // C++: enum HoughModes
    public static final int
            HOUGH_STANDARD = 0,
            HOUGH_PROBABILISTIC = 1,
            HOUGH_MULTI_SCALE = 2,
            HOUGH_GRADIENT = 3,
            HOUGH_GRADIENT_ALT = 4;


    // C++: enum ConnectedComponentsAlgorithmsTypes
    public static final int
            CCL_WU = 0,
            CCL_DEFAULT = -1,
            CCL_GRANA = 1;


    // C++: enum RetrievalModes
    public static final int
            RETR_EXTERNAL = 0,
            RETR_LIST = 1,
            RETR_CCOMP = 2,
            RETR_TREE = 3,
            RETR_FLOODFILL = 4;


    // C++: enum GrabCutClasses
    public static final int
            GC_BGD = 0,
            GC_FGD = 1,
            GC_PR_BGD = 2,
            GC_PR_FGD = 3;


    // C++: enum ColormapTypes
    public static final int
            COLORMAP_AUTUMN = 0,
            COLORMAP_BONE = 1,
            COLORMAP_JET = 2,
            COLORMAP_WINTER = 3,
            COLORMAP_RAINBOW = 4,
            COLORMAP_OCEAN = 5,
            COLORMAP_SUMMER = 6,
            COLORMAP_SPRING = 7,
            COLORMAP_COOL = 8,
            COLORMAP_HSV = 9,
            COLORMAP_PINK = 10,
            COLORMAP_HOT = 11,
            COLORMAP_PARULA = 12,
            COLORMAP_MAGMA = 13,
            COLORMAP_INFERNO = 14,
            COLORMAP_PLASMA = 15,
            COLORMAP_VIRIDIS = 16,
            COLORMAP_CIVIDIS = 17,
            COLORMAP_TWILIGHT = 18,
            COLORMAP_TWILIGHT_SHIFTED = 19,
            COLORMAP_TURBO = 20;


    // C++: enum HistCompMethods
    public static final int
            HISTCMP_CORREL = 0,
            HISTCMP_CHISQR = 1,
            HISTCMP_INTERSECT = 2,
            HISTCMP_BHATTACHARYYA = 3,
            HISTCMP_HELLINGER = HISTCMP_BHATTACHARYYA,
            HISTCMP_CHISQR_ALT = 4,
            HISTCMP_KL_DIV = 5;


    // C++: enum LineTypes
    public static final int
            FILLED = -1,
            LINE_4 = 4,
            LINE_8 = 8,
            LINE_AA = 16;


    // C++: enum InterpolationFlags
    public static final int
            INTER_NEAREST = 0,
            INTER_LINEAR = 1,
            INTER_CUBIC = 2,
            INTER_AREA = 3,
            INTER_LANCZOS4 = 4,
            INTER_LINEAR_EXACT = 5,
            INTER_MAX = 7,
            WARP_FILL_OUTLIERS = 8,
            WARP_INVERSE_MAP = 16;


    // C++: enum SpecialFilter
    public static final int
            FILTER_SCHARR = -1;


    // C++: enum ContourApproximationModes
    public static final int
            CHAIN_APPROX_NONE = 1,
            CHAIN_APPROX_SIMPLE = 2,
            CHAIN_APPROX_TC89_L1 = 3,
            CHAIN_APPROX_TC89_KCOS = 4;


    // C++: enum RectanglesIntersectTypes
    public static final int
            INTERSECT_NONE = 0,
            INTERSECT_PARTIAL = 1,
            INTERSECT_FULL = 2;


    // C++: enum <unnamed>
    public static final int
            CV_GAUSSIAN_5x5 = 7,
            CV_SCHARR = -1,
            CV_MAX_SOBEL_KSIZE = 7,
            CV_RGBA2mRGBA = 125,
            CV_mRGBA2RGBA = 126,
            CV_WARP_FILL_OUTLIERS = 8,
            CV_WARP_INVERSE_MAP = 16,
            CV_CHAIN_CODE = 0,
            CV_LINK_RUNS = 5,
            CV_POLY_APPROX_DP = 0,
            CV_CONTOURS_MATCH_I1 = 1,
            CV_CONTOURS_MATCH_I2 = 2,
            CV_CONTOURS_MATCH_I3 = 3,
            CV_CLOCKWISE = 1,
            CV_COUNTER_CLOCKWISE = 2,
            CV_COMP_CORREL = 0,
            CV_COMP_CHISQR = 1,
            CV_COMP_INTERSECT = 2,
            CV_COMP_BHATTACHARYYA = 3,
            CV_COMP_HELLINGER = CV_COMP_BHATTACHARYYA,
            CV_COMP_CHISQR_ALT = 4,
            CV_COMP_KL_DIV = 5,
            CV_DIST_MASK_3 = 3,
            CV_DIST_MASK_5 = 5,
            CV_DIST_MASK_PRECISE = 0,
            CV_DIST_LABEL_CCOMP = 0,
            CV_DIST_LABEL_PIXEL = 1,
            CV_DIST_USER = -1,
            CV_DIST_L1 = 1,
            CV_DIST_L2 = 2,
            CV_DIST_C = 3,
            CV_DIST_L12 = 4,
            CV_DIST_FAIR = 5,
            CV_DIST_WELSCH = 6,
            CV_DIST_HUBER = 7,
            CV_CANNY_L2_GRADIENT = (1 << 31),
            CV_HOUGH_STANDARD = 0,
            CV_HOUGH_PROBABILISTIC = 1,
            CV_HOUGH_MULTI_SCALE = 2,
            CV_HOUGH_GRADIENT = 3;


    // C++: enum ShapeMatchModes
    public static final int
            CONTOURS_MATCH_I1 = 1,
            CONTOURS_MATCH_I2 = 2,
            CONTOURS_MATCH_I3 = 3;


    // C++: enum WarpPolarMode
    public static final int
            WARP_POLAR_LINEAR = 0,
            WARP_POLAR_LOG = 256;


    // C++: enum ColorConversionCodes
    public static final int
            COLOR_BGR2BGRA = 0,
            COLOR_RGB2RGBA = COLOR_BGR2BGRA,
            COLOR_BGRA2BGR = 1,
            COLOR_RGBA2RGB = COLOR_BGRA2BGR,
            COLOR_BGR2RGBA = 2,
            COLOR_RGB2BGRA = COLOR_BGR2RGBA,
            COLOR_RGBA2BGR = 3,
            COLOR_BGRA2RGB = COLOR_RGBA2BGR,
            COLOR_BGR2RGB = 4,
            COLOR_RGB2BGR = COLOR_BGR2RGB,
            COLOR_BGRA2RGBA = 5,
            COLOR_RGBA2BGRA = COLOR_BGRA2RGBA,
            COLOR_BGR2GRAY = 6,
            COLOR_RGB2GRAY = 7,
            COLOR_GRAY2BGR = 8,
            COLOR_GRAY2RGB = COLOR_GRAY2BGR,
            COLOR_GRAY2BGRA = 9,
            COLOR_GRAY2RGBA = COLOR_GRAY2BGRA,
            COLOR_BGRA2GRAY = 10,
            COLOR_RGBA2GRAY = 11,
            COLOR_BGR2BGR565 = 12,
            COLOR_RGB2BGR565 = 13,
            COLOR_BGR5652BGR = 14,
            COLOR_BGR5652RGB = 15,
            COLOR_BGRA2BGR565 = 16,
            COLOR_RGBA2BGR565 = 17,
            COLOR_BGR5652BGRA = 18,
            COLOR_BGR5652RGBA = 19,
            COLOR_GRAY2BGR565 = 20,
            COLOR_BGR5652GRAY = 21,
            COLOR_BGR2BGR555 = 22,
            COLOR_RGB2BGR555 = 23,
            COLOR_BGR5552BGR = 24,
            COLOR_BGR5552RGB = 25,
            COLOR_BGRA2BGR555 = 26,
            COLOR_RGBA2BGR555 = 27,
            COLOR_BGR5552BGRA = 28,
            COLOR_BGR5552RGBA = 29,
            COLOR_GRAY2BGR555 = 30,
            COLOR_BGR5552GRAY = 31,
            COLOR_BGR2XYZ = 32,
            COLOR_RGB2XYZ = 33,
            COLOR_XYZ2BGR = 34,
            COLOR_XYZ2RGB = 35,
            COLOR_BGR2YCrCb = 36,
            COLOR_RGB2YCrCb = 37,
            COLOR_YCrCb2BGR = 38,
            COLOR_YCrCb2RGB = 39,
            COLOR_BGR2HSV = 40,
            COLOR_RGB2HSV = 41,
            COLOR_BGR2Lab = 44,
            COLOR_RGB2Lab = 45,
            COLOR_BGR2Luv = 50,
            COLOR_RGB2Luv = 51,
            COLOR_BGR2HLS = 52,
            COLOR_RGB2HLS = 53,
            COLOR_HSV2BGR = 54,
            COLOR_HSV2RGB = 55,
            COLOR_Lab2BGR = 56,
            COLOR_Lab2RGB = 57,
            COLOR_Luv2BGR = 58,
            COLOR_Luv2RGB = 59,
            COLOR_HLS2BGR = 60,
            COLOR_HLS2RGB = 61,
            COLOR_BGR2HSV_FULL = 66,
            COLOR_RGB2HSV_FULL = 67,
            COLOR_BGR2HLS_FULL = 68,
            COLOR_RGB2HLS_FULL = 69,
            COLOR_HSV2BGR_FULL = 70,
            COLOR_HSV2RGB_FULL = 71,
            COLOR_HLS2BGR_FULL = 72,
            COLOR_HLS2RGB_FULL = 73,
            COLOR_LBGR2Lab = 74,
            COLOR_LRGB2Lab = 75,
            COLOR_LBGR2Luv = 76,
            COLOR_LRGB2Luv = 77,
            COLOR_Lab2LBGR = 78,
            COLOR_Lab2LRGB = 79,
            COLOR_Luv2LBGR = 80,
            COLOR_Luv2LRGB = 81,
            COLOR_BGR2YUV = 82,
            COLOR_RGB2YUV = 83,
            COLOR_YUV2BGR = 84,
            COLOR_YUV2RGB = 85,
            COLOR_YUV2RGB_NV12 = 90,
            COLOR_YUV2BGR_NV12 = 91,
            COLOR_YUV2RGB_NV21 = 92,
            COLOR_YUV2BGR_NV21 = 93,
            COLOR_YUV420sp2RGB = COLOR_YUV2RGB_NV21,
            COLOR_YUV420sp2BGR = COLOR_YUV2BGR_NV21,
            COLOR_YUV2RGBA_NV12 = 94,
            COLOR_YUV2BGRA_NV12 = 95,
            COLOR_YUV2RGBA_NV21 = 96,
            COLOR_YUV2BGRA_NV21 = 97,
            COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21,
            COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21,
            COLOR_YUV2RGB_YV12 = 98,
            COLOR_YUV2BGR_YV12 = 99,
            COLOR_YUV2RGB_IYUV = 100,
            COLOR_YUV2BGR_IYUV = 101,
            COLOR_YUV2RGB_I420 = COLOR_YUV2RGB_IYUV,
            COLOR_YUV2BGR_I420 = COLOR_YUV2BGR_IYUV,
            COLOR_YUV420p2RGB = COLOR_YUV2RGB_YV12,
            COLOR_YUV420p2BGR = COLOR_YUV2BGR_YV12,
            COLOR_YUV2RGBA_YV12 = 102,
            COLOR_YUV2BGRA_YV12 = 103,
            COLOR_YUV2RGBA_IYUV = 104,
            COLOR_YUV2BGRA_IYUV = 105,
            COLOR_YUV2RGBA_I420 = COLOR_YUV2RGBA_IYUV,
            COLOR_YUV2BGRA_I420 = COLOR_YUV2BGRA_IYUV,
            COLOR_YUV420p2RGBA = COLOR_YUV2RGBA_YV12,
            COLOR_YUV420p2BGRA = COLOR_YUV2BGRA_YV12,
            COLOR_YUV2GRAY_420 = 106,
            COLOR_YUV2GRAY_NV21 = COLOR_YUV2GRAY_420,
            COLOR_YUV2GRAY_NV12 = COLOR_YUV2GRAY_420,
            COLOR_YUV2GRAY_YV12 = COLOR_YUV2GRAY_420,
            COLOR_YUV2GRAY_IYUV = COLOR_YUV2GRAY_420,
            COLOR_YUV2GRAY_I420 = COLOR_YUV2GRAY_420,
            COLOR_YUV420sp2GRAY = COLOR_YUV2GRAY_420,
            COLOR_YUV420p2GRAY = COLOR_YUV2GRAY_420,
            COLOR_YUV2RGB_UYVY = 107,
            COLOR_YUV2BGR_UYVY = 108,
            COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY,
            COLOR_YUV2BGR_Y422 = COLOR_YUV2BGR_UYVY,
            COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY,
            COLOR_YUV2BGR_UYNV = COLOR_YUV2BGR_UYVY,
            COLOR_YUV2RGBA_UYVY = 111,
            COLOR_YUV2BGRA_UYVY = 112,
            COLOR_YUV2RGBA_Y422 = COLOR_YUV2RGBA_UYVY,
            COLOR_YUV2BGRA_Y422 = COLOR_YUV2BGRA_UYVY,
            COLOR_YUV2RGBA_UYNV = COLOR_YUV2RGBA_UYVY,
            COLOR_YUV2BGRA_UYNV = COLOR_YUV2BGRA_UYVY,
            COLOR_YUV2RGB_YUY2 = 115,
            COLOR_YUV2BGR_YUY2 = 116,
            COLOR_YUV2RGB_YVYU = 117,
            COLOR_YUV2BGR_YVYU = 118,
            COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2,
            COLOR_YUV2BGR_YUYV = COLOR_YUV2BGR_YUY2,
            COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2,
            COLOR_YUV2BGR_YUNV = COLOR_YUV2BGR_YUY2,
            COLOR_YUV2RGBA_YUY2 = 119,
            COLOR_YUV2BGRA_YUY2 = 120,
            COLOR_YUV2RGBA_YVYU = 121,
            COLOR_YUV2BGRA_YVYU = 122,
            COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2,
            COLOR_YUV2BGRA_YUYV = COLOR_YUV2BGRA_YUY2,
            COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2,
            COLOR_YUV2BGRA_YUNV = COLOR_YUV2BGRA_YUY2,
            COLOR_YUV2GRAY_UYVY = 123,
            COLOR_YUV2GRAY_YUY2 = 124,
            COLOR_YUV2GRAY_Y422 = COLOR_YUV2GRAY_UYVY,
            COLOR_YUV2GRAY_UYNV = COLOR_YUV2GRAY_UYVY,
            COLOR_YUV2GRAY_YVYU = COLOR_YUV2GRAY_YUY2,
            COLOR_YUV2GRAY_YUYV = COLOR_YUV2GRAY_YUY2,
            COLOR_YUV2GRAY_YUNV = COLOR_YUV2GRAY_YUY2,
            COLOR_RGBA2mRGBA = 125,
            COLOR_mRGBA2RGBA = 126,
            COLOR_RGB2YUV_I420 = 127,
            COLOR_BGR2YUV_I420 = 128,
            COLOR_RGB2YUV_IYUV = COLOR_RGB2YUV_I420,
            COLOR_BGR2YUV_IYUV = COLOR_BGR2YUV_I420,
            COLOR_RGBA2YUV_I420 = 129,
            COLOR_BGRA2YUV_I420 = 130,
            COLOR_RGBA2YUV_IYUV = COLOR_RGBA2YUV_I420,
            COLOR_BGRA2YUV_IYUV = COLOR_BGRA2YUV_I420,
            COLOR_RGB2YUV_YV12 = 131,
            COLOR_BGR2YUV_YV12 = 132,
            COLOR_RGBA2YUV_YV12 = 133,
            COLOR_BGRA2YUV_YV12 = 134,
            COLOR_BayerBG2BGR = 46,
            COLOR_BayerGB2BGR = 47,
            COLOR_BayerRG2BGR = 48,
            COLOR_BayerGR2BGR = 49,
            COLOR_BayerBG2RGB = COLOR_BayerRG2BGR,
            COLOR_BayerGB2RGB = COLOR_BayerGR2BGR,
            COLOR_BayerRG2RGB = COLOR_BayerBG2BGR,
            COLOR_BayerGR2RGB = COLOR_BayerGB2BGR,
            COLOR_BayerBG2GRAY = 86,
            COLOR_BayerGB2GRAY = 87,
            COLOR_BayerRG2GRAY = 88,
            COLOR_BayerGR2GRAY = 89,
            COLOR_BayerBG2BGR_VNG = 62,
            COLOR_BayerGB2BGR_VNG = 63,
            COLOR_BayerRG2BGR_VNG = 64,
            COLOR_BayerGR2BGR_VNG = 65,
            COLOR_BayerBG2RGB_VNG = COLOR_BayerRG2BGR_VNG,
            COLOR_BayerGB2RGB_VNG = COLOR_BayerGR2BGR_VNG,
            COLOR_BayerRG2RGB_VNG = COLOR_BayerBG2BGR_VNG,
            COLOR_BayerGR2RGB_VNG = COLOR_BayerGB2BGR_VNG,
            COLOR_BayerBG2BGR_EA = 135,
            COLOR_BayerGB2BGR_EA = 136,
            COLOR_BayerRG2BGR_EA = 137,
            COLOR_BayerGR2BGR_EA = 138,
            COLOR_BayerBG2RGB_EA = COLOR_BayerRG2BGR_EA,
            COLOR_BayerGB2RGB_EA = COLOR_BayerGR2BGR_EA,
            COLOR_BayerRG2RGB_EA = COLOR_BayerBG2BGR_EA,
            COLOR_BayerGR2RGB_EA = COLOR_BayerGB2BGR_EA,
            COLOR_BayerBG2BGRA = 139,
            COLOR_BayerGB2BGRA = 140,
            COLOR_BayerRG2BGRA = 141,
            COLOR_BayerGR2BGRA = 142,
            COLOR_BayerBG2RGBA = COLOR_BayerRG2BGRA,
            COLOR_BayerGB2RGBA = COLOR_BayerGR2BGRA,
            COLOR_BayerRG2RGBA = COLOR_BayerBG2BGRA,
            COLOR_BayerGR2RGBA = COLOR_BayerGB2BGRA,
            COLOR_COLORCVT_MAX = 143;


    // C++: enum LineSegmentDetectorModes
    public static final int
            LSD_REFINE_NONE = 0,
            LSD_REFINE_STD = 1,
            LSD_REFINE_ADV = 2;


    // C++: enum ThresholdTypes
    public static final int
            THRESH_BINARY = 0,
            THRESH_BINARY_INV = 1,
            THRESH_TRUNC = 2,
            THRESH_TOZERO = 3,
            THRESH_TOZERO_INV = 4,
            THRESH_MASK = 7,
            THRESH_OTSU = 8,
            THRESH_TRIANGLE = 16;


    // C++: enum AdaptiveThresholdTypes
    public static final int
            ADAPTIVE_THRESH_MEAN_C = 0,
            ADAPTIVE_THRESH_GAUSSIAN_C = 1;


    // C++: enum MorphShapes_c
    public static final int
            CV_SHAPE_RECT = 0,
            CV_SHAPE_CROSS = 1,
            CV_SHAPE_ELLIPSE = 2,
            CV_SHAPE_CUSTOM = 100;


    // C++: enum GrabCutModes
    public static final int
            GC_INIT_WITH_RECT = 0,
            GC_INIT_WITH_MASK = 1,
            GC_EVAL = 2,
            GC_EVAL_FREEZE_MODEL = 3;


    // C++: enum MorphShapes
    public static final int
            MORPH_RECT = 0,
            MORPH_CROSS = 1,
            MORPH_ELLIPSE = 2;


    // C++: enum DistanceTransformLabelTypes
    public static final int
            DIST_LABEL_CCOMP = 0,
            DIST_LABEL_PIXEL = 1;


    // C++: enum DistanceTypes
    public static final int
            DIST_USER = -1,
            DIST_L1 = 1,
            DIST_L2 = 2,
            DIST_C = 3,
            DIST_L12 = 4,
            DIST_FAIR = 5,
            DIST_WELSCH = 6,
            DIST_HUBER = 7;


    // C++: enum TemplateMatchModes
    public static final int
            TM_SQDIFF = 0,
            TM_SQDIFF_NORMED = 1,
            TM_CCORR = 2,
            TM_CCORR_NORMED = 3,
            TM_CCOEFF = 4,
            TM_CCOEFF_NORMED = 5;


    // C++: enum DistanceTransformMasks
    public static final int
            DIST_MASK_3 = 3,
            DIST_MASK_5 = 5,
            DIST_MASK_PRECISE = 0;


    // C++: enum ConnectedComponentsTypes
    public static final int
            CC_STAT_LEFT = 0,
            CC_STAT_TOP = 1,
            CC_STAT_WIDTH = 2,
            CC_STAT_HEIGHT = 3,
            CC_STAT_AREA = 4,
            CC_STAT_MAX = 5;


    // C++: enum SmoothMethod_c
    public static final int
            CV_BLUR_NO_SCALE = 0,
            CV_BLUR = 1,
            CV_GAUSSIAN = 2,
            CV_MEDIAN = 3,
            CV_BILATERAL = 4;


    // C++: enum MarkerTypes
    public static final int
            MARKER_CROSS = 0,
            MARKER_TILTED_CROSS = 1,
            MARKER_STAR = 2,
            MARKER_DIAMOND = 3,
            MARKER_SQUARE = 4,
            MARKER_TRIANGLE_UP = 5,
            MARKER_TRIANGLE_DOWN = 6;


    //
    // C++:  Mat cv::getAffineTransform(vector_Point2f src, vector_Point2f dst)
    //

    public static Mat getAffineTransform(MatOfPoint2f src, MatOfPoint2f dst) {
        Mat src_mat = src;
        Mat dst_mat = dst;
        return new Mat(getAffineTransform_0(src_mat.nativeObj, dst_mat.nativeObj));
    }


    //
    // C++:  Mat cv::getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi = CV_PI*0.5, int ktype = CV_64F)
    //

    /**
     * Returns Gabor filter coefficients.
     *
     * For more details about gabor filter equations and parameters, see: [Gabor
     * Filter](http://en.wikipedia.org/wiki/Gabor_filter).
     *
     * @param ksize Size of the filter returned.
     * @param sigma Standard deviation of the gaussian envelope.
     * @param theta Orientation of the normal to the parallel stripes of a Gabor function.
     * @param lambd Wavelength of the sinusoidal factor.
     * @param gamma Spatial aspect ratio.
     * @param psi Phase offset.
     * @param ktype Type of filter coefficients. It can be CV_32F or CV_64F .
     * @return automatically generated
     */
    public static Mat getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi, int ktype) {
        return new Mat(getGaborKernel_0(ksize.width, ksize.height, sigma, theta, lambd, gamma, psi, ktype));
    }

    /**
     * Returns Gabor filter coefficients.
     *
     * For more details about gabor filter equations and parameters, see: [Gabor
     * Filter](http://en.wikipedia.org/wiki/Gabor_filter).
     *
     * @param ksize Size of the filter returned.
     * @param sigma Standard deviation of the gaussian envelope.
     * @param theta Orientation of the normal to the parallel stripes of a Gabor function.
     * @param lambd Wavelength of the sinusoidal factor.
     * @param gamma Spatial aspect ratio.
     * @param psi Phase offset.
     * @return automatically generated
     */
    public static Mat getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi) {
        return new Mat(getGaborKernel_1(ksize.width, ksize.height, sigma, theta, lambd, gamma, psi));
    }

    /**
     * Returns Gabor filter coefficients.
     *
     * For more details about gabor filter equations and parameters, see: [Gabor
     * Filter](http://en.wikipedia.org/wiki/Gabor_filter).
     *
     * @param ksize Size of the filter returned.
     * @param sigma Standard deviation of the gaussian envelope.
     * @param theta Orientation of the normal to the parallel stripes of a Gabor function.
     * @param lambd Wavelength of the sinusoidal factor.
     * @param gamma Spatial aspect ratio.
     * @return automatically generated
     */
    public static Mat getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma) {
        return new Mat(getGaborKernel_2(ksize.width, ksize.height, sigma, theta, lambd, gamma));
    }


    //
    // C++:  Mat cv::getGaussianKernel(int ksize, double sigma, int ktype = CV_64F)
    //

    /**
     * Returns Gaussian filter coefficients.
     *
     * The function computes and returns the \(\texttt{ksize} \times 1\) matrix of Gaussian filter
     * coefficients:
     *
     * \(G_i= \alpha *e^{-(i-( \texttt{ksize} -1)/2)^2/(2* \texttt{sigma}^2)},\)
     *
     * where \(i=0..\texttt{ksize}-1\) and \(\alpha\) is the scale factor chosen so that \(\sum_i G_i=1\).
     *
     * Two of such generated kernels can be passed to sepFilter2D. Those functions automatically recognize
     * smoothing kernels (a symmetrical kernel with sum of weights equal to 1) and handle them accordingly.
     * You may also use the higher-level GaussianBlur.
     * @param ksize Aperture size. It should be odd ( \(\texttt{ksize} \mod 2 = 1\) ) and positive.
     * @param sigma Gaussian standard deviation. If it is non-positive, it is computed from ksize as
     * {@code sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8}.
     * @param ktype Type of filter coefficients. It can be CV_32F or CV_64F .
     * SEE:  sepFilter2D, getDerivKernels, getStructuringElement, GaussianBlur
     * @return automatically generated
     */
    public static Mat getGaussianKernel(int ksize, double sigma, int ktype) {
        return new Mat(getGaussianKernel_0(ksize, sigma, ktype));
    }

    /**
     * Returns Gaussian filter coefficients.
     *
     * The function computes and returns the \(\texttt{ksize} \times 1\) matrix of Gaussian filter
     * coefficients:
     *
     * \(G_i= \alpha *e^{-(i-( \texttt{ksize} -1)/2)^2/(2* \texttt{sigma}^2)},\)
     *
     * where \(i=0..\texttt{ksize}-1\) and \(\alpha\) is the scale factor chosen so that \(\sum_i G_i=1\).
     *
     * Two of such generated kernels can be passed to sepFilter2D. Those functions automatically recognize
     * smoothing kernels (a symmetrical kernel with sum of weights equal to 1) and handle them accordingly.
     * You may also use the higher-level GaussianBlur.
     * @param ksize Aperture size. It should be odd ( \(\texttt{ksize} \mod 2 = 1\) ) and positive.
     * @param sigma Gaussian standard deviation. If it is non-positive, it is computed from ksize as
     * {@code sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8}.
     * SEE:  sepFilter2D, getDerivKernels, getStructuringElement, GaussianBlur
     * @return automatically generated
     */
    public static Mat getGaussianKernel(int ksize, double sigma) {
        return new Mat(getGaussianKernel_1(ksize, sigma));
    }


    //
    // C++:  Mat cv::getPerspectiveTransform(Mat src, Mat dst, int solveMethod = DECOMP_LU)
    //

    /**
     * Calculates a perspective transform from four pairs of the corresponding points.
     *
     * The function calculates the \(3 \times 3\) matrix of a perspective transform so that:
     *
     * \(\begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}\)
     *
     * where
     *
     * \(dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2,3\)
     *
     * @param src Coordinates of quadrangle vertices in the source image.
     * @param dst Coordinates of the corresponding quadrangle vertices in the destination image.
     * @param solveMethod method passed to cv::solve (#DecompTypes)
     *
     * SEE:  findHomography, warpPerspective, perspectiveTransform
     * @return automatically generated
     */
    public static Mat getPerspectiveTransform(Mat src, Mat dst, int solveMethod) {
        return new Mat(getPerspectiveTransform_0(src.nativeObj, dst.nativeObj, solveMethod));
    }

    /**
     * Calculates a perspective transform from four pairs of the corresponding points.
     *
     * The function calculates the \(3 \times 3\) matrix of a perspective transform so that:
     *
     * \(\begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}\)
     *
     * where
     *
     * \(dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2,3\)
     *
     * @param src Coordinates of quadrangle vertices in the source image.
     * @param dst Coordinates of the corresponding quadrangle vertices in the destination image.
     *
     * SEE:  findHomography, warpPerspective, perspectiveTransform
     * @return automatically generated
     */
    public static Mat getPerspectiveTransform(Mat src, Mat dst) {
        return new Mat(getPerspectiveTransform_1(src.nativeObj, dst.nativeObj));
    }


    //
    // C++:  Mat cv::getRotationMatrix2D(Point2f center, double angle, double scale)
    //

    /**
     * Calculates an affine matrix of 2D rotation.
     *
     * The function calculates the following matrix:
     *
     * \(\begin{bmatrix} \alpha &amp;  \beta &amp; (1- \alpha )  \cdot \texttt{center.x} -  \beta \cdot \texttt{center.y} \\ - \beta &amp;  \alpha &amp;  \beta \cdot \texttt{center.x} + (1- \alpha )  \cdot \texttt{center.y} \end{bmatrix}\)
     *
     * where
     *
     * \(\begin{array}{l} \alpha =  \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta =  \texttt{scale} \cdot \sin \texttt{angle} \end{array}\)
     *
     * The transformation maps the rotation center to itself. If this is not the target, adjust the shift.
     *
     * @param center Center of the rotation in the source image.
     * @param angle Rotation angle in degrees. Positive values mean counter-clockwise rotation (the
     * coordinate origin is assumed to be the top-left corner).
     * @param scale Isotropic scale factor.
     *
     * SEE:  getAffineTransform, warpAffine, transform
     * @return automatically generated
     */
    public static Mat getRotationMatrix2D(Point center, double angle, double scale) {
        return new Mat(getRotationMatrix2D_0(center.x, center.y, angle, scale));
    }


    //
    // C++:  Mat cv::getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1))
    //

    /**
     * Returns a structuring element of the specified size and shape for morphological operations.
     *
     * The function constructs and returns the structuring element that can be further passed to #erode,
     * #dilate or #morphologyEx. But you can also construct an arbitrary binary mask yourself and use it as
     * the structuring element.
     *
     * @param shape Element shape that could be one of #MorphShapes
     * @param ksize Size of the structuring element.
     * @param anchor Anchor position within the element. The default value \((-1, -1)\) means that the
     * anchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor
     * position. In other cases the anchor just regulates how much the result of the morphological
     * operation is shifted.
     * @return automatically generated
     */
    public static Mat getStructuringElement(int shape, Size ksize, Point anchor) {
        return new Mat(getStructuringElement_0(shape, ksize.width, ksize.height, anchor.x, anchor.y));
    }

    /**
     * Returns a structuring element of the specified size and shape for morphological operations.
     *
     * The function constructs and returns the structuring element that can be further passed to #erode,
     * #dilate or #morphologyEx. But you can also construct an arbitrary binary mask yourself and use it as
     * the structuring element.
     *
     * @param shape Element shape that could be one of #MorphShapes
     * @param ksize Size of the structuring element.
     * anchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor
     * position. In other cases the anchor just regulates how much the result of the morphological
     * operation is shifted.
     * @return automatically generated
     */
    public static Mat getStructuringElement(int shape, Size ksize) {
        return new Mat(getStructuringElement_1(shape, ksize.width, ksize.height));
    }


    //
    // C++:  Moments cv::moments(Mat array, bool binaryImage = false)
    //

    /**
     * Calculates all of the moments up to the third order of a polygon or rasterized shape.
     *
     * The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape. The
     * results are returned in the structure cv::Moments.
     *
     * @param array Raster image (single-channel, 8-bit or floating-point 2D array) or an array (
     * \(1 \times N\) or \(N \times 1\) ) of 2D points (Point or Point2f ).
     * @param binaryImage If it is true, all non-zero image pixels are treated as 1's. The parameter is
     * used for images only.
     * @return moments.
     *
     * <b>Note:</b> Only applicable to contour moments calculations from Python bindings: Note that the numpy
     * type for the input array should be either np.int32 or np.float32.
     *
     * SEE:  contourArea, arcLength
     */
    public static Moments moments(Mat array, boolean binaryImage) {
        return new Moments(moments_0(array.nativeObj, binaryImage));
    }

    /**
     * Calculates all of the moments up to the third order of a polygon or rasterized shape.
     *
     * The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape. The
     * results are returned in the structure cv::Moments.
     *
     * @param array Raster image (single-channel, 8-bit or floating-point 2D array) or an array (
     * \(1 \times N\) or \(N \times 1\) ) of 2D points (Point or Point2f ).
     * used for images only.
     * @return moments.
     *
     * <b>Note:</b> Only applicable to contour moments calculations from Python bindings: Note that the numpy
     * type for the input array should be either np.int32 or np.float32.
     *
     * SEE:  contourArea, arcLength
     */
    public static Moments moments(Mat array) {
        return new Moments(moments_1(array.nativeObj));
    }


    //
    // C++:  Point2d cv::phaseCorrelate(Mat src1, Mat src2, Mat window = Mat(), double* response = 0)
    //

    /**
     * The function is used to detect translational shifts that occur between two images.
     *
     * The operation takes advantage of the Fourier shift theorem for detecting the translational shift in
     * the frequency domain. It can be used for fast image registration as well as motion estimation. For
     * more information please see &lt;http://en.wikipedia.org/wiki/Phase_correlation&gt;
     *
     * Calculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed
     * with getOptimalDFTSize.
     *
     * The function performs the following equations:
     * <ul>
     *   <li>
     *  First it applies a Hanning window (see &lt;http://en.wikipedia.org/wiki/Hann_function&gt;) to each
     * image to remove possible edge effects. This window is cached until the array size changes to speed
     * up processing time.
     *   </li>
     *   <li>
     *  Next it computes the forward DFTs of each source array:
     * \(\mathbf{G}_a = \mathcal{F}\{src_1\}, \; \mathbf{G}_b = \mathcal{F}\{src_2\}\)
     * where \(\mathcal{F}\) is the forward DFT.
     *   </li>
     *   <li>
     *  It then computes the cross-power spectrum of each frequency domain array:
     * \(R = \frac{ \mathbf{G}_a \mathbf{G}_b^*}{|\mathbf{G}_a \mathbf{G}_b^*|}\)
     *   </li>
     *   <li>
     *  Next the cross-correlation is converted back into the time domain via the inverse DFT:
     * \(r = \mathcal{F}^{-1}\{R\}\)
     *   </li>
     *   <li>
     *  Finally, it computes the peak location and computes a 5x5 weighted centroid around the peak to
     * achieve sub-pixel accuracy.
     * \((\Delta x, \Delta y) = \texttt{weightedCentroid} \{\arg \max_{(x, y)}\{r\}\}\)
     *   </li>
     *   <li>
     *  If non-zero, the response parameter is computed as the sum of the elements of r within the 5x5
     * centroid around the peak location. It is normalized to a maximum of 1 (meaning there is a single
     * peak) and will be smaller when there are multiple peaks.
     *   </li>
     * </ul>
     *
     * @param src1 Source floating point array (CV_32FC1 or CV_64FC1)
     * @param src2 Source floating point array (CV_32FC1 or CV_64FC1)
     * @param window Floating point array with windowing coefficients to reduce edge effects (optional).
     * @param response Signal power within the 5x5 centroid around the peak, between 0 and 1 (optional).
     * @return detected phase shift (sub-pixel) between the two arrays.
     *
     * SEE: dft, getOptimalDFTSize, idft, mulSpectrums createHanningWindow
     */
    public static Point phaseCorrelate(Mat src1, Mat src2, Mat window, double[] response) {
        double[] response_out = new double[1];
        Point retVal = new Point(phaseCorrelate_0(src1.nativeObj, src2.nativeObj, window.nativeObj, response_out));
        if(response!=null) response[0] = (double)response_out[0];
        return retVal;
    }

    /**
     * The function is used to detect translational shifts that occur between two images.
     *
     * The operation takes advantage of the Fourier shift theorem for detecting the translational shift in
     * the frequency domain. It can be used for fast image registration as well as motion estimation. For
     * more information please see &lt;http://en.wikipedia.org/wiki/Phase_correlation&gt;
     *
     * Calculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed
     * with getOptimalDFTSize.
     *
     * The function performs the following equations:
     * <ul>
     *   <li>
     *  First it applies a Hanning window (see &lt;http://en.wikipedia.org/wiki/Hann_function&gt;) to each
     * image to remove possible edge effects. This window is cached until the array size changes to speed
     * up processing time.
     *   </li>
     *   <li>
     *  Next it computes the forward DFTs of each source array:
     * \(\mathbf{G}_a = \mathcal{F}\{src_1\}, \; \mathbf{G}_b = \mathcal{F}\{src_2\}\)
     * where \(\mathcal{F}\) is the forward DFT.
     *   </li>
     *   <li>
     *  It then computes the cross-power spectrum of each frequency domain array:
     * \(R = \frac{ \mathbf{G}_a \mathbf{G}_b^*}{|\mathbf{G}_a \mathbf{G}_b^*|}\)
     *   </li>
     *   <li>
     *  Next the cross-correlation is converted back into the time domain via the inverse DFT:
     * \(r = \mathcal{F}^{-1}\{R\}\)
     *   </li>
     *   <li>
     *  Finally, it computes the peak location and computes a 5x5 weighted centroid around the peak to
     * achieve sub-pixel accuracy.
     * \((\Delta x, \Delta y) = \texttt{weightedCentroid} \{\arg \max_{(x, y)}\{r\}\}\)
     *   </li>
     *   <li>
     *  If non-zero, the response parameter is computed as the sum of the elements of r within the 5x5
     * centroid around the peak location. It is normalized to a maximum of 1 (meaning there is a single
     * peak) and will be smaller when there are multiple peaks.
     *   </li>
     * </ul>
     *
     * @param src1 Source floating point array (CV_32FC1 or CV_64FC1)
     * @param src2 Source floating point array (CV_32FC1 or CV_64FC1)
     * @param window Floating point array with windowing coefficients to reduce edge effects (optional).
     * @return detected phase shift (sub-pixel) between the two arrays.
     *
     * SEE: dft, getOptimalDFTSize, idft, mulSpectrums createHanningWindow
     */
    public static Point phaseCorrelate(Mat src1, Mat src2, Mat window) {
        return new Point(phaseCorrelate_1(src1.nativeObj, src2.nativeObj, window.nativeObj));
    }

    /**
     * The function is used to detect translational shifts that occur between two images.
     *
     * The operation takes advantage of the Fourier shift theorem for detecting the translational shift in
     * the frequency domain. It can be used for fast image registration as well as motion estimation. For
     * more information please see &lt;http://en.wikipedia.org/wiki/Phase_correlation&gt;
     *
     * Calculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed
     * with getOptimalDFTSize.
     *
     * The function performs the following equations:
     * <ul>
     *   <li>
     *  First it applies a Hanning window (see &lt;http://en.wikipedia.org/wiki/Hann_function&gt;) to each
     * image to remove possible edge effects. This window is cached until the array size changes to speed
     * up processing time.
     *   </li>
     *   <li>
     *  Next it computes the forward DFTs of each source array:
     * \(\mathbf{G}_a = \mathcal{F}\{src_1\}, \; \mathbf{G}_b = \mathcal{F}\{src_2\}\)
     * where \(\mathcal{F}\) is the forward DFT.
     *   </li>
     *   <li>
     *  It then computes the cross-power spectrum of each frequency domain array:
     * \(R = \frac{ \mathbf{G}_a \mathbf{G}_b^*}{|\mathbf{G}_a \mathbf{G}_b^*|}\)
     *   </li>
     *   <li>
     *  Next the cross-correlation is converted back into the time domain via the inverse DFT:
     * \(r = \mathcal{F}^{-1}\{R\}\)
     *   </li>
     *   <li>
     *  Finally, it computes the peak location and computes a 5x5 weighted centroid around the peak to
     * achieve sub-pixel accuracy.
     * \((\Delta x, \Delta y) = \texttt{weightedCentroid} \{\arg \max_{(x, y)}\{r\}\}\)
     *   </li>
     *   <li>
     *  If non-zero, the response parameter is computed as the sum of the elements of r within the 5x5
     * centroid around the peak location. It is normalized to a maximum of 1 (meaning there is a single
     * peak) and will be smaller when there are multiple peaks.
     *   </li>
     * </ul>
     *
     * @param src1 Source floating point array (CV_32FC1 or CV_64FC1)
     * @param src2 Source floating point array (CV_32FC1 or CV_64FC1)
     * @return detected phase shift (sub-pixel) between the two arrays.
     *
     * SEE: dft, getOptimalDFTSize, idft, mulSpectrums createHanningWindow
     */
    public static Point phaseCorrelate(Mat src1, Mat src2) {
        return new Point(phaseCorrelate_2(src1.nativeObj, src2.nativeObj));
    }


    //
    // C++:  Ptr_CLAHE cv::createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8))
    //

    /**
     * Creates a smart pointer to a cv::CLAHE class and initializes it.
     *
     * @param clipLimit Threshold for contrast limiting.
     * @param tileGridSize Size of grid for histogram equalization. Input image will be divided into
     * equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.
     * @return automatically generated
     */
    public static CLAHE createCLAHE(double clipLimit, Size tileGridSize) {
        return CLAHE.__fromPtr__(createCLAHE_0(clipLimit, tileGridSize.width, tileGridSize.height));
    }

    /**
     * Creates a smart pointer to a cv::CLAHE class and initializes it.
     *
     * @param clipLimit Threshold for contrast limiting.
     * equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.
     * @return automatically generated
     */
    public static CLAHE createCLAHE(double clipLimit) {
        return CLAHE.__fromPtr__(createCLAHE_1(clipLimit));
    }

    /**
     * Creates a smart pointer to a cv::CLAHE class and initializes it.
     *
     * equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.
     * @return automatically generated
     */
    public static CLAHE createCLAHE() {
        return CLAHE.__fromPtr__(createCLAHE_2());
    }


    //
    // C++:  Ptr_GeneralizedHoughBallard cv::createGeneralizedHoughBallard()
    //

    /**
     * Creates a smart pointer to a cv::GeneralizedHoughBallard class and initializes it.
     * @return automatically generated
     */
    public static GeneralizedHoughBallard createGeneralizedHoughBallard() {
        return GeneralizedHoughBallard.__fromPtr__(createGeneralizedHoughBallard_0());
    }


    //
    // C++:  Ptr_GeneralizedHoughGuil cv::createGeneralizedHoughGuil()
    //

    /**
     * Creates a smart pointer to a cv::GeneralizedHoughGuil class and initializes it.
     * @return automatically generated
     */
    public static GeneralizedHoughGuil createGeneralizedHoughGuil() {
        return GeneralizedHoughGuil.__fromPtr__(createGeneralizedHoughGuil_0());
    }


    //
    // C++:  Ptr_LineSegmentDetector cv::createLineSegmentDetector(int _refine = LSD_REFINE_STD, double _scale = 0.8, double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5, double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024)
    //

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
     * @param _scale The scale of the image that will be used to find the lines. Range (0..1].
     * @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
     * @param _quant Bound to the quantization error on the gradient norm.
     * @param _ang_th Gradient angle tolerance in degrees.
     * @param _log_eps Detection threshold: -log10(NFA) &gt; log_eps. Used only when advance refinement
     * is chosen.
     * @param _density_th Minimal density of aligned region points in the enclosing rectangle.
     * @param _n_bins Number of bins in pseudo-ordering of gradient modulus.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th, double _log_eps, double _density_th, int _n_bins) {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_0(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th, _n_bins));
    }

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
     * @param _scale The scale of the image that will be used to find the lines. Range (0..1].
     * @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
     * @param _quant Bound to the quantization error on the gradient norm.
     * @param _ang_th Gradient angle tolerance in degrees.
     * @param _log_eps Detection threshold: -log10(NFA) &gt; log_eps. Used only when advance refinement
     * is chosen.
     * @param _density_th Minimal density of aligned region points in the enclosing rectangle.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th, double _log_eps, double _density_th) {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_1(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th));
    }

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
     * @param _scale The scale of the image that will be used to find the lines. Range (0..1].
     * @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
     * @param _quant Bound to the quantization error on the gradient norm.
     * @param _ang_th Gradient angle tolerance in degrees.
     * @param _log_eps Detection threshold: -log10(NFA) &gt; log_eps. Used only when advance refinement
     * is chosen.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th, double _log_eps) {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_2(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps));
    }

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
     * @param _scale The scale of the image that will be used to find the lines. Range (0..1].
     * @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
     * @param _quant Bound to the quantization error on the gradient norm.
     * @param _ang_th Gradient angle tolerance in degrees.
     * is chosen.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th) {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_3(_refine, _scale, _sigma_scale, _quant, _ang_th));
    }

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
     * @param _scale The scale of the image that will be used to find the lines. Range (0..1].
     * @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
     * @param _quant Bound to the quantization error on the gradient norm.
     * is chosen.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector(int _refine, double _scale, double _sigma_scale, double _quant) {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_4(_refine, _scale, _sigma_scale, _quant));
    }

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
     * @param _scale The scale of the image that will be used to find the lines. Range (0..1].
     * @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
     * is chosen.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector(int _refine, double _scale, double _sigma_scale) {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_5(_refine, _scale, _sigma_scale));
    }

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
     * @param _scale The scale of the image that will be used to find the lines. Range (0..1].
     * is chosen.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector(int _refine, double _scale) {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_6(_refine, _scale));
    }

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
     * is chosen.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector(int _refine) {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_7(_refine));
    }

    /**
     * Creates a smart pointer to a LineSegmentDetector object and initializes it.
     *
     * The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
     * to edit those, as to tailor it for their own application.
     *
     * is chosen.
     *
     * <b>Note:</b> Implementation has been removed due original code license conflict
     * @return automatically generated
     */
    public static LineSegmentDetector createLineSegmentDetector() {
        return LineSegmentDetector.__fromPtr__(createLineSegmentDetector_8());
    }


    //
    // C++:  Rect cv::boundingRect(Mat array)
    //

    /**
     * Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
     *
     * The function calculates and returns the minimal up-right bounding rectangle for the specified point set or
     * non-zero pixels of gray-scale image.
     *
     * @param array Input gray-scale image or 2D point set, stored in std::vector or Mat.
     * @return automatically generated
     */
    public static Rect boundingRect(Mat array) {
        return new Rect(boundingRect_0(array.nativeObj));
    }


    //
    // C++:  RotatedRect cv::fitEllipse(vector_Point2f points)
    //

    /**
     * Fits an ellipse around a set of 2D points.
     *
     * The function calculates the ellipse that fits (in a least-squares sense) a set of 2D points best of
     * all. It returns the rotated rectangle in which the ellipse is inscribed. The first algorithm described by CITE: Fitzgibbon95
     * is used. Developer should keep in mind that it is possible that the returned
     * ellipse/rotatedRect data contains negative indices, due to the data points being close to the
     * border of the containing Mat element.
     *
     * @param points Input 2D point set, stored in std::vector&lt;&gt; or Mat
     * @return automatically generated
     */
    public static RotatedRect fitEllipse(MatOfPoint2f points) {
        Mat points_mat = points;
        return new RotatedRect(fitEllipse_0(points_mat.nativeObj));
    }


    //
    // C++:  RotatedRect cv::fitEllipseAMS(Mat points)
    //

    /**
     * Fits an ellipse around a set of 2D points.
     *
     *  The function calculates the ellipse that fits a set of 2D points.
     *  It returns the rotated rectangle in which the ellipse is inscribed.
     *  The Approximate Mean Square (AMS) proposed by CITE: Taubin1991 is used.
     *
     *  For an ellipse, this basis set is \( \chi= \left(x^2, x y, y^2, x, y, 1\right) \),
     *  which is a set of six free coefficients \( A^T=\left\{A_{\text{xx}},A_{\text{xy}},A_{\text{yy}},A_x,A_y,A_0\right\} \).
     *  However, to specify an ellipse, all that is needed is five numbers; the major and minor axes lengths \( (a,b) \),
     *  the position \( (x_0,y_0) \), and the orientation \( \theta \). This is because the basis set includes lines,
     *  quadratics, parabolic and hyperbolic functions as well as elliptical functions as possible fits.
     *  If the fit is found to be a parabolic or hyperbolic function then the standard #fitEllipse method is used.
     *  The AMS method restricts the fit to parabolic, hyperbolic and elliptical curves
     *  by imposing the condition that \( A^T ( D_x^T D_x  +   D_y^T D_y) A = 1 \) where
     *  the matrices \( Dx \) and \( Dy \) are the partial derivatives of the design matrix \( D \) with
     *  respect to x and y. The matrices are formed row by row applying the following to
     *  each of the points in the set:
     *  \(align*}{
     *  D(i,:)&amp;=\left\{x_i^2, x_i y_i, y_i^2, x_i, y_i, 1\right\} &amp;
     *  D_x(i,:)&amp;=\left\{2 x_i,y_i,0,1,0,0\right\} &amp;
     *  D_y(i,:)&amp;=\left\{0,x_i,2 y_i,0,1,0\right\}
     *  \)
     *  The AMS method minimizes the cost function
     *  \(equation*}{
     *  \epsilon ^2=\frac{ A^T D^T D A }{ A^T (D_x^T D_x +  D_y^T D_y) A^T }
     *  \)
     *
     *  The minimum cost is found by solving the generalized eigenvalue problem.
     *
     *  \(equation*}{
     *  D^T D A = \lambda  \left( D_x^T D_x +  D_y^T D_y\right) A
     *  \)
     *
     *  @param points Input 2D point set, stored in std::vector&lt;&gt; or Mat
     * @return automatically generated
     */
    public static RotatedRect fitEllipseAMS(Mat points) {
        return new RotatedRect(fitEllipseAMS_0(points.nativeObj));
    }


    //
    // C++:  RotatedRect cv::fitEllipseDirect(Mat points)
    //

    /**
     * Fits an ellipse around a set of 2D points.
     *
     *  The function calculates the ellipse that fits a set of 2D points.
     *  It returns the rotated rectangle in which the ellipse is inscribed.
     *  The Direct least square (Direct) method by CITE: Fitzgibbon1999 is used.
     *
     *  For an ellipse, this basis set is \( \chi= \left(x^2, x y, y^2, x, y, 1\right) \),
     *  which is a set of six free coefficients \( A^T=\left\{A_{\text{xx}},A_{\text{xy}},A_{\text{yy}},A_x,A_y,A_0\right\} \).
     *  However, to specify an ellipse, all that is needed is five numbers; the major and minor axes lengths \( (a,b) \),
     *  the position \( (x_0,y_0) \), and the orientation \( \theta \). This is because the basis set includes lines,
     *  quadratics, parabolic and hyperbolic functions as well as elliptical functions as possible fits.
     *  The Direct method confines the fit to ellipses by ensuring that \( 4 A_{xx} A_{yy}- A_{xy}^2 &gt; 0 \).
     *  The condition imposed is that \( 4 A_{xx} A_{yy}- A_{xy}^2=1 \) which satisfies the inequality
     *  and as the coefficients can be arbitrarily scaled is not overly restrictive.
     *
     *  \(equation*}{
     *  \epsilon ^2= A^T D^T D A \quad \text{with} \quad A^T C A =1 \quad \text{and} \quad C=\left(\begin{matrix}
     *  0 &amp; 0  &amp; 2  &amp; 0  &amp; 0  &amp;  0  \\
     *  0 &amp; -1  &amp; 0  &amp; 0  &amp; 0  &amp;  0 \\
     *  2 &amp; 0  &amp; 0  &amp; 0  &amp; 0  &amp;  0 \\
     *  0 &amp; 0  &amp; 0  &amp; 0  &amp; 0  &amp;  0 \\
     *  0 &amp; 0  &amp; 0  &amp; 0  &amp; 0  &amp;  0 \\
     *  0 &amp; 0  &amp; 0  &amp; 0  &amp; 0  &amp;  0
     *  \end{matrix} \right)
     *  \)
     *
     *  The minimum cost is found by solving the generalized eigenvalue problem.
     *
     *  \(equation*}{
     *  D^T D A = \lambda  \left( C\right) A
     *  \)
     *
     *  The system produces only one positive eigenvalue \( \lambda\) which is chosen as the solution
     *  with its eigenvector \(\mathbf{u}\). These are used to find the coefficients
     *
     *  \(equation*}{
     *  A = \sqrt{\frac{1}{\mathbf{u}^T C \mathbf{u}}}  \mathbf{u}
     *  \)
     *  The scaling factor guarantees that  \(A^T C A =1\).
     *
     *  @param points Input 2D point set, stored in std::vector&lt;&gt; or Mat
     * @return automatically generated
     */
    public static RotatedRect fitEllipseDirect(Mat points) {
        return new RotatedRect(fitEllipseDirect_0(points.nativeObj));
    }


    //
    // C++:  RotatedRect cv::minAreaRect(vector_Point2f points)
    //

    /**
     * Finds a rotated rectangle of the minimum area enclosing the input 2D point set.
     *
     * The function calculates and returns the minimum-area bounding rectangle (possibly rotated) for a
     * specified point set. Developer should keep in mind that the returned RotatedRect can contain negative
     * indices when data is close to the containing Mat element boundary.
     *
     * @param points Input vector of 2D points, stored in std::vector&lt;&gt; or Mat
     * @return automatically generated
     */
    public static RotatedRect minAreaRect(MatOfPoint2f points) {
        Mat points_mat = points;
        return new RotatedRect(minAreaRect_0(points_mat.nativeObj));
    }


    //
    // C++:  bool cv::clipLine(Rect imgRect, Point& pt1, Point& pt2)
    //

    /**
     *
     * @param imgRect Image rectangle.
     * @param pt1 First line point.
     * @param pt2 Second line point.
     * @return automatically generated
     */
    public static boolean clipLine(Rect imgRect, Point pt1, Point pt2) {
        double[] pt1_out = new double[2];
        double[] pt2_out = new double[2];
        boolean retVal = clipLine_0(imgRect.x, imgRect.y, imgRect.width, imgRect.height, pt1.x, pt1.y, pt1_out, pt2.x, pt2.y, pt2_out);
        if(pt1!=null){ pt1.x = pt1_out[0]; pt1.y = pt1_out[1]; } 
        if(pt2!=null){ pt2.x = pt2_out[0]; pt2.y = pt2_out[1]; } 
        return retVal;
    }


    //
    // C++:  bool cv::isContourConvex(vector_Point contour)
    //

    /**
     * Tests a contour convexity.
     *
     * The function tests whether the input contour is convex or not. The contour must be simple, that is,
     * without self-intersections. Otherwise, the function output is undefined.
     *
     * @param contour Input vector of 2D points, stored in std::vector&lt;&gt; or Mat
     * @return automatically generated
     */
    public static boolean isContourConvex(MatOfPoint contour) {
        Mat contour_mat = contour;
        return isContourConvex_0(contour_mat.nativeObj);
    }


    //
    // C++:  double cv::arcLength(vector_Point2f curve, bool closed)
    //

    /**
     * Calculates a contour perimeter or a curve length.
     *
     * The function computes a curve length or a closed contour perimeter.
     *
     * @param curve Input vector of 2D points, stored in std::vector or Mat.
     * @param closed Flag indicating whether the curve is closed or not.
     * @return automatically generated
     */
    public static double arcLength(MatOfPoint2f curve, boolean closed) {
        Mat curve_mat = curve;
        return arcLength_0(curve_mat.nativeObj, closed);
    }


    //
    // C++:  double cv::compareHist(Mat H1, Mat H2, int method)
    //

    /**
     * Compares two histograms.
     *
     * The function cv::compareHist compares two dense or two sparse histograms using the specified method.
     *
     * The function returns \(d(H_1, H_2)\) .
     *
     * While the function works well with 1-, 2-, 3-dimensional dense histograms, it may not be suitable
     * for high-dimensional sparse histograms. In such histograms, because of aliasing and sampling
     * problems, the coordinates of non-zero histogram bins can slightly shift. To compare such histograms
     * or more general sparse configurations of weighted points, consider using the #EMD function.
     *
     * @param H1 First compared histogram.
     * @param H2 Second compared histogram of the same size as H1 .
     * @param method Comparison method, see #HistCompMethods
     * @return automatically generated
     */
    public static double compareHist(Mat H1, Mat H2, int method) {
        return compareHist_0(H1.nativeObj, H2.nativeObj, method);
    }


    //
    // C++:  double cv::contourArea(Mat contour, bool oriented = false)
    //

    /**
     * Calculates a contour area.
     *
     * The function computes a contour area. Similarly to moments , the area is computed using the Green
     * formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using
     * #drawContours or #fillPoly , can be different. Also, the function will most certainly give a wrong
     * results for contours with self-intersections.
     *
     * Example:
     * <code>
     *     vector&lt;Point&gt; contour;
     *     contour.push_back(Point2f(0, 0));
     *     contour.push_back(Point2f(10, 0));
     *     contour.push_back(Point2f(10, 10));
     *     contour.push_back(Point2f(5, 4));
     *
     *     double area0 = contourArea(contour);
     *     vector&lt;Point&gt; approx;
     *     approxPolyDP(contour, approx, 5, true);
     *     double area1 = contourArea(approx);
     *
     *     cout &lt;&lt; "area0 =" &lt;&lt; area0 &lt;&lt; endl &lt;&lt;
     *             "area1 =" &lt;&lt; area1 &lt;&lt; endl &lt;&lt;
     *             "approx poly vertices" &lt;&lt; approx.size() &lt;&lt; endl;
     * </code>
     * @param contour Input vector of 2D points (contour vertices), stored in std::vector or Mat.
     * @param oriented Oriented area flag. If it is true, the function returns a signed area value,
     * depending on the contour orientation (clockwise or counter-clockwise). Using this feature you can
     * determine orientation of a contour by taking the sign of an area. By default, the parameter is
     * false, which means that the absolute value is returned.
     * @return automatically generated
     */
    public static double contourArea(Mat contour, boolean oriented) {
        return contourArea_0(contour.nativeObj, oriented);
    }

    /**
     * Calculates a contour area.
     *
     * The function computes a contour area. Similarly to moments , the area is computed using the Green
     * formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using
     * #drawContours or #fillPoly , can be different. Also, the function will most certainly give a wrong
     * results for contours with self-intersections.
     *
     * Example:
     * <code>
     *     vector&lt;Point&gt; contour;
     *     contour.push_back(Point2f(0, 0));
     *     contour.push_back(Point2f(10, 0));
     *     contour.push_back(Point2f(10, 10));
     *     contour.push_back(Point2f(5, 4));
     *
     *     double area0 = contourArea(contour);
     *     vector&lt;Point&gt; approx;
     *     approxPolyDP(contour, approx, 5, true);
     *     double area1 = contourArea(approx);
     *
     *     cout &lt;&lt; "area0 =" &lt;&lt; area0 &lt;&lt; endl &lt;&lt;
     *             "area1 =" &lt;&lt; area1 &lt;&lt; endl &lt;&lt;
     *             "approx poly vertices" &lt;&lt; approx.size() &lt;&lt; endl;
     * </code>
     * @param contour Input vector of 2D points (contour vertices), stored in std::vector or Mat.
     * depending on the contour orientation (clockwise or counter-clockwise). Using this feature you can
     * determine orientation of a contour by taking the sign of an area. By default, the parameter is
     * false, which means that the absolute value is returned.
     * @return automatically generated
     */
    public static double contourArea(Mat contour) {
        return contourArea_1(contour.nativeObj);
    }


    //
    // C++:  double cv::getFontScaleFromHeight(int fontFace, int pixelHeight, int thickness = 1)
    //

    /**
     * Calculates the font-specific size to use to achieve a given height in pixels.
     *
     * @param fontFace Font to use, see cv::HersheyFonts.
     * @param pixelHeight Pixel height to compute the fontScale for
     * @param thickness Thickness of lines used to render the text.See putText for details.
     * @return The fontSize to use for cv::putText
     *
     * SEE: cv::putText
     */
    public static double getFontScaleFromHeight(int fontFace, int pixelHeight, int thickness) {
        return getFontScaleFromHeight_0(fontFace, pixelHeight, thickness);
    }

    /**
     * Calculates the font-specific size to use to achieve a given height in pixels.
     *
     * @param fontFace Font to use, see cv::HersheyFonts.
     * @param pixelHeight Pixel height to compute the fontScale for
     * @return The fontSize to use for cv::putText
     *
     * SEE: cv::putText
     */
    public static double getFontScaleFromHeight(int fontFace, int pixelHeight) {
        return getFontScaleFromHeight_1(fontFace, pixelHeight);
    }


    //
    // C++:  double cv::matchShapes(Mat contour1, Mat contour2, int method, double parameter)
    //

    /**
     * Compares two shapes.
     *
     * The function compares two shapes. All three implemented methods use the Hu invariants (see #HuMoments)
     *
     * @param contour1 First contour or grayscale image.
     * @param contour2 Second contour or grayscale image.
     * @param method Comparison method, see #ShapeMatchModes
     * @param parameter Method-specific parameter (not supported now).
     * @return automatically generated
     */
    public static double matchShapes(Mat contour1, Mat contour2, int method, double parameter) {
        return matchShapes_0(contour1.nativeObj, contour2.nativeObj, method, parameter);
    }


    //
    // C++:  double cv::minEnclosingTriangle(Mat points, Mat& triangle)
    //

    /**
     * Finds a triangle of minimum area enclosing a 2D point set and returns its area.
     *
     * The function finds a triangle of minimum area enclosing the given set of 2D points and returns its
     * area. The output for a given 2D point set is shown in the image below. 2D points are depicted in
     * red* and the enclosing triangle in *yellow*.
     *
     * ![Sample output of the minimum enclosing triangle function](pics/minenclosingtriangle.png)
     *
     * The implementation of the algorithm is based on O'Rourke's CITE: ORourke86 and Klee and Laskowski's
     * CITE: KleeLaskowski85 papers. O'Rourke provides a \(\theta(n)\) algorithm for finding the minimal
     * enclosing triangle of a 2D convex polygon with n vertices. Since the #minEnclosingTriangle function
     * takes a 2D point set as input an additional preprocessing step of computing the convex hull of the
     * 2D point set is required. The complexity of the #convexHull function is \(O(n log(n))\) which is higher
     * than \(\theta(n)\). Thus the overall complexity of the function is \(O(n log(n))\).
     *
     * @param points Input vector of 2D points with depth CV_32S or CV_32F, stored in std::vector&lt;&gt; or Mat
     * @param triangle Output vector of three 2D points defining the vertices of the triangle. The depth
     * of the OutputArray must be CV_32F.
     * @return automatically generated
     */
    public static double minEnclosingTriangle(Mat points, Mat triangle) {
        return minEnclosingTriangle_0(points.nativeObj, triangle.nativeObj);
    }


    //
    // C++:  double cv::pointPolygonTest(vector_Point2f contour, Point2f pt, bool measureDist)
    //

    /**
     * Performs a point-in-contour test.
     *
     * The function determines whether the point is inside a contour, outside, or lies on an edge (or
     * coincides with a vertex). It returns positive (inside), negative (outside), or zero (on an edge)
     * value, correspondingly. When measureDist=false , the return value is +1, -1, and 0, respectively.
     * Otherwise, the return value is a signed distance between the point and the nearest contour edge.
     *
     * See below a sample output of the function where each image pixel is tested against the contour:
     *
     * ![sample output](pics/pointpolygon.png)
     *
     * @param contour Input contour.
     * @param pt Point tested against the contour.
     * @param measureDist If true, the function estimates the signed distance from the point to the
     * nearest contour edge. Otherwise, the function only checks if the point is inside a contour or not.
     * @return automatically generated
     */
    public static double pointPolygonTest(MatOfPoint2f contour, Point pt, boolean measureDist) {
        Mat contour_mat = contour;
        return pointPolygonTest_0(contour_mat.nativeObj, pt.x, pt.y, measureDist);
    }


    //
    // C++:  double cv::threshold(Mat src, Mat& dst, double thresh, double maxval, int type)
    //

    /**
     * Applies a fixed-level threshold to each array element.
     *
     * The function applies fixed-level thresholding to a multiple-channel array. The function is typically
     * used to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for
     * this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
     * values. There are several types of thresholding supported by the function. They are determined by
     * type parameter.
     *
     * Also, the special values #THRESH_OTSU or #THRESH_TRIANGLE may be combined with one of the
     * above values. In these cases, the function determines the optimal threshold value using the Otsu's
     * or Triangle algorithm and uses it instead of the specified thresh.
     *
     * <b>Note:</b> Currently, the Otsu's and Triangle methods are implemented only for 8-bit single-channel images.
     *
     * @param src input array (multiple-channel, 8-bit or 32-bit floating point).
     * @param dst output array of the same size  and type and the same number of channels as src.
     * @param thresh threshold value.
     * @param maxval maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholding
     * types.
     * @param type thresholding type (see #ThresholdTypes).
     * @return the computed threshold value if Otsu's or Triangle methods used.
     *
     * SEE:  adaptiveThreshold, findContours, compare, min, max
     */
    public static double threshold(Mat src, Mat dst, double thresh, double maxval, int type) {
        return threshold_0(src.nativeObj, dst.nativeObj, thresh, maxval, type);
    }


    //
    // C++:  float cv::intersectConvexConvex(Mat _p1, Mat _p2, Mat& _p12, bool handleNested = true)
    //

    /**
     * Finds intersection of two convex polygons
     *
     * @param _p1 First polygon
     * @param _p2 Second polygon
     * @param _p12 Output polygon describing the intersecting area
     * @param handleNested When true, an intersection is found if one of the polygons is fully enclosed in the other.
     * When false, no intersection is found. If the polygons share a side or the vertex of one polygon lies on an edge
     * of the other, they are not considered nested and an intersection will be found regardless of the value of handleNested.
     *
     * @return Absolute value of area of intersecting polygon
     *
     * <b>Note:</b> intersectConvexConvex doesn't confirm that both polygons are convex and will return invalid results if they aren't.
     */
    public static float intersectConvexConvex(Mat _p1, Mat _p2, Mat _p12, boolean handleNested) {
        return intersectConvexConvex_0(_p1.nativeObj, _p2.nativeObj, _p12.nativeObj, handleNested);
    }

    /**
     * Finds intersection of two convex polygons
     *
     * @param _p1 First polygon
     * @param _p2 Second polygon
     * @param _p12 Output polygon describing the intersecting area
     * When false, no intersection is found. If the polygons share a side or the vertex of one polygon lies on an edge
     * of the other, they are not considered nested and an intersection will be found regardless of the value of handleNested.
     *
     * @return Absolute value of area of intersecting polygon
     *
     * <b>Note:</b> intersectConvexConvex doesn't confirm that both polygons are convex and will return invalid results if they aren't.
     */
    public static float intersectConvexConvex(Mat _p1, Mat _p2, Mat _p12) {
        return intersectConvexConvex_1(_p1.nativeObj, _p2.nativeObj, _p12.nativeObj);
    }


    //
    // C++:  float cv::wrapperEMD(Mat signature1, Mat signature2, int distType, Mat cost = Mat(), Ptr_float& lowerBound = Ptr<float>(), Mat& flow = Mat())
    //

    /**
     * Computes the "minimal work" distance between two weighted point configurations.
     *
     * The function computes the earth mover distance and/or a lower boundary of the distance between the
     * two weighted point configurations. One of the applications described in CITE: RubnerSept98,
     * CITE: Rubner2000 is multi-dimensional histogram comparison for image retrieval. EMD is a transportation
     * problem that is solved using some modification of a simplex algorithm, thus the complexity is
     * exponential in the worst case, though, on average it is much faster. In the case of a real metric
     * the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used
     * to determine roughly whether the two signatures are far enough so that they cannot relate to the
     * same object.
     *
     * @param signature1 First signature, a \(\texttt{size1}\times \texttt{dims}+1\) floating-point matrix.
     * Each row stores the point weight followed by the point coordinates. The matrix is allowed to have
     * a single column (weights only) if the user-defined cost matrix is used. The weights must be
     * non-negative and have at least one non-zero value.
     * @param signature2 Second signature of the same format as signature1 , though the number of rows
     * may be different. The total weights may be different. In this case an extra "dummy" point is added
     * to either signature1 or signature2. The weights must be non-negative and have at least one non-zero
     * value.
     * @param distType Used metric. See #DistanceTypes.
     * @param cost User-defined \(\texttt{size1}\times \texttt{size2}\) cost matrix. Also, if a cost matrix
     * is used, lower boundary lowerBound cannot be calculated because it needs a metric function.
     * signatures that is a distance between mass centers. The lower boundary may not be calculated if
     * the user-defined cost matrix is used, the total weights of point configurations are not equal, or
     * if the signatures consist of weights only (the signature matrices have a single column). You
     * <b>must</b> initialize \*lowerBound . If the calculated distance between mass centers is greater or
     * equal to \*lowerBound (it means that the signatures are far enough), the function does not
     * calculate EMD. In any case \*lowerBound is set to the calculated distance between mass centers on
     * return. Thus, if you want to calculate both distance between mass centers and EMD, \*lowerBound
     * should be set to 0.
     * @param flow Resultant \(\texttt{size1} \times \texttt{size2}\) flow matrix: \(\texttt{flow}_{i,j}\) is
     * a flow from \(i\) -th point of signature1 to \(j\) -th point of signature2 .
     * @return automatically generated
     */
    public static float EMD(Mat signature1, Mat signature2, int distType, Mat cost, Mat flow) {
        return EMD_0(signature1.nativeObj, signature2.nativeObj, distType, cost.nativeObj, flow.nativeObj);
    }

    /**
     * Computes the "minimal work" distance between two weighted point configurations.
     *
     * The function computes the earth mover distance and/or a lower boundary of the distance between the
     * two weighted point configurations. One of the applications described in CITE: RubnerSept98,
     * CITE: Rubner2000 is multi-dimensional histogram comparison for image retrieval. EMD is a transportation
     * problem that is solved using some modification of a simplex algorithm, thus the complexity is
     * exponential in the worst case, though, on average it is much faster. In the case of a real metric
     * the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used
     * to determine roughly whether the two signatures are far enough so that they cannot relate to the
     * same object.
     *
     * @param signature1 First signature, a \(\texttt{size1}\times \texttt{dims}+1\) floating-point matrix.
     * Each row stores the point weight followed by the point coordinates. The matrix is allowed to have
     * a single column (weights only) if the user-defined cost matrix is used. The weights must be
     * non-negative and have at least one non-zero value.
     * @param signature2 Second signature of the same format as signature1 , though the number of rows
     * may be different. The total weights may be different. In this case an extra "dummy" point is added
     * to either signature1 or signature2. The weights must be non-negative and have at least one non-zero
     * value.
     * @param distType Used metric. See #DistanceTypes.
     * @param cost User-defined \(\texttt{size1}\times \texttt{size2}\) cost matrix. Also, if a cost matrix
     * is used, lower boundary lowerBound cannot be calculated because it needs a metric function.
     * signatures that is a distance between mass centers. The lower boundary may not be calculated if
     * the user-defined cost matrix is used, the total weights of point configurations are not equal, or
     * if the signatures consist of weights only (the signature matrices have a single column). You
     * <b>must</b> initialize \*lowerBound . If the calculated distance between mass centers is greater or
     * equal to \*lowerBound (it means that the signatures are far enough), the function does not
     * calculate EMD. In any case \*lowerBound is set to the calculated distance between mass centers on
     * return. Thus, if you want to calculate both distance between mass centers and EMD, \*lowerBound
     * should be set to 0.
     * a flow from \(i\) -th point of signature1 to \(j\) -th point of signature2 .
     * @return automatically generated
     */
    public static float EMD(Mat signature1, Mat signature2, int distType, Mat cost) {
        return EMD_1(signature1.nativeObj, signature2.nativeObj, distType, cost.nativeObj);
    }

    /**
     * Computes the "minimal work" distance between two weighted point configurations.
     *
     * The function computes the earth mover distance and/or a lower boundary of the distance between the
     * two weighted point configurations. One of the applications described in CITE: RubnerSept98,
     * CITE: Rubner2000 is multi-dimensional histogram comparison for image retrieval. EMD is a transportation
     * problem that is solved using some modification of a simplex algorithm, thus the complexity is
     * exponential in the worst case, though, on average it is much faster. In the case of a real metric
     * the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used
     * to determine roughly whether the two signatures are far enough so that they cannot relate to the
     * same object.
     *
     * @param signature1 First signature, a \(\texttt{size1}\times \texttt{dims}+1\) floating-point matrix.
     * Each row stores the point weight followed by the point coordinates. The matrix is allowed to have
     * a single column (weights only) if the user-defined cost matrix is used. The weights must be
     * non-negative and have at least one non-zero value.
     * @param signature2 Second signature of the same format as signature1 , though the number of rows
     * may be different. The total weights may be different. In this case an extra "dummy" point is added
     * to either signature1 or signature2. The weights must be non-negative and have at least one non-zero
     * value.
     * @param distType Used metric. See #DistanceTypes.
     * is used, lower boundary lowerBound cannot be calculated because it needs a metric function.
     * signatures that is a distance between mass centers. The lower boundary may not be calculated if
     * the user-defined cost matrix is used, the total weights of point configurations are not equal, or
     * if the signatures consist of weights only (the signature matrices have a single column). You
     * <b>must</b> initialize \*lowerBound . If the calculated distance between mass centers is greater or
     * equal to \*lowerBound (it means that the signatures are far enough), the function does not
     * calculate EMD. In any case \*lowerBound is set to the calculated distance between mass centers on
     * return. Thus, if you want to calculate both distance between mass centers and EMD, \*lowerBound
     * should be set to 0.
     * a flow from \(i\) -th point of signature1 to \(j\) -th point of signature2 .
     * @return automatically generated
     */
    public static float EMD(Mat signature1, Mat signature2, int distType) {
        return EMD_3(signature1.nativeObj, signature2.nativeObj, distType);
    }


    //
    // C++:  int cv::connectedComponents(Mat image, Mat& labels, int connectivity, int ltype, int ccltype)
    //

    /**
     * computes the connected components labeled image of boolean image
     *
     * image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0
     * represents the background label. ltype specifies the output label image type, an important
     * consideration based on the total number of labels or alternatively the total number of pixels in
     * the source image. ccltype specifies the connected components labeling algorithm to use, currently
     * Grana (BBDT) and Wu's (SAUF) algorithms are supported, see the #ConnectedComponentsAlgorithmsTypes
     * for details. Note that SAUF algorithm forces a row major ordering of labels while BBDT does not.
     * This function uses parallel version of both Grana and Wu's algorithms if at least one allowed
     * parallel framework is enabled and if the rows of the image are at least twice the number returned by #getNumberOfCPUs.
     *
     * @param image the 8-bit single-channel image to be labeled
     * @param labels destination labeled image
     * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
     * @param ltype output image label type. Currently CV_32S and CV_16U are supported.
     * @param ccltype connected components algorithm type (see the #ConnectedComponentsAlgorithmsTypes).
     * @return automatically generated
     */
    public static int connectedComponentsWithAlgorithm(Mat image, Mat labels, int connectivity, int ltype, int ccltype) {
        return connectedComponentsWithAlgorithm_0(image.nativeObj, labels.nativeObj, connectivity, ltype, ccltype);
    }


    //
    // C++:  int cv::connectedComponents(Mat image, Mat& labels, int connectivity = 8, int ltype = CV_32S)
    //

    /**
     *
     *
     * @param image the 8-bit single-channel image to be labeled
     * @param labels destination labeled image
     * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
     * @param ltype output image label type. Currently CV_32S and CV_16U are supported.
     * @return automatically generated
     */
    public static int connectedComponents(Mat image, Mat labels, int connectivity, int ltype) {
        return connectedComponents_0(image.nativeObj, labels.nativeObj, connectivity, ltype);
    }

    /**
     *
     *
     * @param image the 8-bit single-channel image to be labeled
     * @param labels destination labeled image
     * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
     * @return automatically generated
     */
    public static int connectedComponents(Mat image, Mat labels, int connectivity) {
        return connectedComponents_1(image.nativeObj, labels.nativeObj, connectivity);
    }

    /**
     *
     *
     * @param image the 8-bit single-channel image to be labeled
     * @param labels destination labeled image
     * @return automatically generated
     */
    public static int connectedComponents(Mat image, Mat labels) {
        return connectedComponents_2(image.nativeObj, labels.nativeObj);
    }


    //
    // C++:  int cv::connectedComponentsWithStats(Mat image, Mat& labels, Mat& stats, Mat& centroids, int connectivity, int ltype, int ccltype)
    //

    /**
     * computes the connected components labeled image of boolean image and also produces a statistics output for each label
     *
     * image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0
     * represents the background label. ltype specifies the output label image type, an important
     * consideration based on the total number of labels or alternatively the total number of pixels in
     * the source image. ccltype specifies the connected components labeling algorithm to use, currently
     * Grana's (BBDT) and Wu's (SAUF) algorithms are supported, see the #ConnectedComponentsAlgorithmsTypes
     * for details. Note that SAUF algorithm forces a row major ordering of labels while BBDT does not.
     * This function uses parallel version of both Grana and Wu's algorithms (statistics included) if at least one allowed
     * parallel framework is enabled and if the rows of the image are at least twice the number returned by #getNumberOfCPUs.
     *
     * @param image the 8-bit single-channel image to be labeled
     * @param labels destination labeled image
     * @param stats statistics output for each label, including the background label, see below for
     * available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
     * #ConnectedComponentsTypes. The data type is CV_32S.
     * @param centroids centroid output for each label, including the background label. Centroids are
     * accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
     * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
     * @param ltype output image label type. Currently CV_32S and CV_16U are supported.
     * @param ccltype connected components algorithm type (see #ConnectedComponentsAlgorithmsTypes).
     * @return automatically generated
     */
    public static int connectedComponentsWithStatsWithAlgorithm(Mat image, Mat labels, Mat stats, Mat centroids, int connectivity, int ltype, int ccltype) {
        return connectedComponentsWithStatsWithAlgorithm_0(image.nativeObj, labels.nativeObj, stats.nativeObj, centroids.nativeObj, connectivity, ltype, ccltype);
    }


    //
    // C++:  int cv::connectedComponentsWithStats(Mat image, Mat& labels, Mat& stats, Mat& centroids, int connectivity = 8, int ltype = CV_32S)
    //

    /**
     *
     * @param image the 8-bit single-channel image to be labeled
     * @param labels destination labeled image
     * @param stats statistics output for each label, including the background label, see below for
     * available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
     * #ConnectedComponentsTypes. The data type is CV_32S.
     * @param centroids centroid output for each label, including the background label. Centroids are
     * accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
     * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
     * @param ltype output image label type. Currently CV_32S and CV_16U are supported.
     * @return automatically generated
     */
    public static int connectedComponentsWithStats(Mat image, Mat labels, Mat stats, Mat centroids, int connectivity, int ltype) {
        return connectedComponentsWithStats_0(image.nativeObj, labels.nativeObj, stats.nativeObj, centroids.nativeObj, connectivity, ltype);
    }

    /**
     *
     * @param image the 8-bit single-channel image to be labeled
     * @param labels destination labeled image
     * @param stats statistics output for each label, including the background label, see below for
     * available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
     * #ConnectedComponentsTypes. The data type is CV_32S.
     * @param centroids centroid output for each label, including the background label. Centroids are
     * accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
     * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
     * @return automatically generated
     */
    public static int connectedComponentsWithStats(Mat image, Mat labels, Mat stats, Mat centroids, int connectivity) {
        return connectedComponentsWithStats_1(image.nativeObj, labels.nativeObj, stats.nativeObj, centroids.nativeObj, connectivity);
    }

    /**
     *
     * @param image the 8-bit single-channel image to be labeled
     * @param labels destination labeled image
     * @param stats statistics output for each label, including the background label, see below for
     * available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
     * #ConnectedComponentsTypes. The data type is CV_32S.
     * @param centroids centroid output for each label, including the background label. Centroids are
     * accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
     * @return automatically generated
     */
    public static int connectedComponentsWithStats(Mat image, Mat labels, Mat stats, Mat centroids) {
        return connectedComponentsWithStats_2(image.nativeObj, labels.nativeObj, stats.nativeObj, centroids.nativeObj);
    }


    //
    // C++:  int cv::floodFill(Mat& image, Mat& mask, Point seedPoint, Scalar newVal, Rect* rect = 0, Scalar loDiff = Scalar(), Scalar upDiff = Scalar(), int flags = 4)
    //

    /**
     * Fills a connected component with the given color.
     *
     * The function cv::floodFill fills a connected component starting from the seed point with the specified
     * color. The connectivity is determined by the color/brightness closeness of the neighbor pixels. The
     * pixel at \((x,y)\) is considered to belong to the repainted domain if:
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and floating range
     * \(\texttt{src} (x',y')- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} (x',y')+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and floating range
     * \(\texttt{src} (x',y')_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} (x',y')_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} (x',y')_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} (x',y')_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} (x',y')_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} (x',y')_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * where \(src(x',y')\) is the value of one of pixel neighbors that is already known to belong to the
     * component. That is, to be added to the connected component, a color/brightness of the pixel should
     * be close enough to:
     * <ul>
     *   <li>
     *  Color/brightness of one of its neighbors that already belong to the connected component in case
     * of a floating range.
     *   </li>
     *   <li>
     *  Color/brightness of the seed point in case of a fixed range.
     *   </li>
     * </ul>
     *
     * Use these functions to either mark a connected component with the specified color in-place, or build
     * a mask and then extract the contour, or copy the region to another image, and so on.
     *
     * @param image Input/output 1- or 3-channel, 8-bit, or floating-point image. It is modified by the
     * function unless the #FLOODFILL_MASK_ONLY flag is set in the second variant of the function. See
     * the details below.
     * @param mask Operation mask that should be a single-channel 8-bit image, 2 pixels wider and 2 pixels
     * taller than image. Since this is both an input and output parameter, you must take responsibility
     * of initializing it. Flood-filling cannot go across non-zero pixels in the input mask. For example,
     * an edge detector output can be used as a mask to stop filling at edges. On output, pixels in the
     * mask corresponding to filled pixels in the image are set to 1 or to the a value specified in flags
     * as described below. Additionally, the function fills the border of the mask with ones to simplify
     * internal processing. It is therefore possible to use the same mask in multiple calls to the function
     * to make sure the filled areas do not overlap.
     * @param seedPoint Starting point.
     * @param newVal New value of the repainted domain pixels.
     * @param loDiff Maximal lower brightness/color difference between the currently observed pixel and
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * @param upDiff Maximal upper brightness/color difference between the currently observed pixel and
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * @param rect Optional output parameter set by the function to the minimum bounding rectangle of the
     * repainted domain.
     * @param flags Operation flags. The first 8 bits contain a connectivity value. The default value of
     * 4 means that only the four nearest neighbor pixels (those that share an edge) are considered. A
     * connectivity value of 8 means that the eight nearest neighbor pixels (those that share a corner)
     * will be considered. The next 8 bits (8-16) contain a value between 1 and 255 with which to fill
     * the mask (the default value is 1). For example, 4 | ( 255 &lt;&lt; 8 ) will consider 4 nearest
     * neighbours and fill the mask with a value of 255. The following additional options occupy higher
     * bits and therefore may be further combined with the connectivity and mask fill values using
     * bit-wise or (|), see #FloodFillFlags.
     *
     * <b>Note:</b> Since the mask is larger than the filled image, a pixel \((x, y)\) in image corresponds to the
     * pixel \((x+1, y+1)\) in the mask .
     *
     * SEE: findContours
     * @return automatically generated
     */
    public static int floodFill(Mat image, Mat mask, Point seedPoint, Scalar newVal, Rect rect, Scalar loDiff, Scalar upDiff, int flags) {
        double[] rect_out = new double[4];
        int retVal = floodFill_0(image.nativeObj, mask.nativeObj, seedPoint.x, seedPoint.y, newVal.val[0], newVal.val[1], newVal.val[2], newVal.val[3], rect_out, loDiff.val[0], loDiff.val[1], loDiff.val[2], loDiff.val[3], upDiff.val[0], upDiff.val[1], upDiff.val[2], upDiff.val[3], flags);
        if(rect!=null){ rect.x = (int)rect_out[0]; rect.y = (int)rect_out[1]; rect.width = (int)rect_out[2]; rect.height = (int)rect_out[3]; } 
        return retVal;
    }

    /**
     * Fills a connected component with the given color.
     *
     * The function cv::floodFill fills a connected component starting from the seed point with the specified
     * color. The connectivity is determined by the color/brightness closeness of the neighbor pixels. The
     * pixel at \((x,y)\) is considered to belong to the repainted domain if:
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and floating range
     * \(\texttt{src} (x',y')- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} (x',y')+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and floating range
     * \(\texttt{src} (x',y')_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} (x',y')_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} (x',y')_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} (x',y')_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} (x',y')_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} (x',y')_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * where \(src(x',y')\) is the value of one of pixel neighbors that is already known to belong to the
     * component. That is, to be added to the connected component, a color/brightness of the pixel should
     * be close enough to:
     * <ul>
     *   <li>
     *  Color/brightness of one of its neighbors that already belong to the connected component in case
     * of a floating range.
     *   </li>
     *   <li>
     *  Color/brightness of the seed point in case of a fixed range.
     *   </li>
     * </ul>
     *
     * Use these functions to either mark a connected component with the specified color in-place, or build
     * a mask and then extract the contour, or copy the region to another image, and so on.
     *
     * @param image Input/output 1- or 3-channel, 8-bit, or floating-point image. It is modified by the
     * function unless the #FLOODFILL_MASK_ONLY flag is set in the second variant of the function. See
     * the details below.
     * @param mask Operation mask that should be a single-channel 8-bit image, 2 pixels wider and 2 pixels
     * taller than image. Since this is both an input and output parameter, you must take responsibility
     * of initializing it. Flood-filling cannot go across non-zero pixels in the input mask. For example,
     * an edge detector output can be used as a mask to stop filling at edges. On output, pixels in the
     * mask corresponding to filled pixels in the image are set to 1 or to the a value specified in flags
     * as described below. Additionally, the function fills the border of the mask with ones to simplify
     * internal processing. It is therefore possible to use the same mask in multiple calls to the function
     * to make sure the filled areas do not overlap.
     * @param seedPoint Starting point.
     * @param newVal New value of the repainted domain pixels.
     * @param loDiff Maximal lower brightness/color difference between the currently observed pixel and
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * @param upDiff Maximal upper brightness/color difference between the currently observed pixel and
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * @param rect Optional output parameter set by the function to the minimum bounding rectangle of the
     * repainted domain.
     * 4 means that only the four nearest neighbor pixels (those that share an edge) are considered. A
     * connectivity value of 8 means that the eight nearest neighbor pixels (those that share a corner)
     * will be considered. The next 8 bits (8-16) contain a value between 1 and 255 with which to fill
     * the mask (the default value is 1). For example, 4 | ( 255 &lt;&lt; 8 ) will consider 4 nearest
     * neighbours and fill the mask with a value of 255. The following additional options occupy higher
     * bits and therefore may be further combined with the connectivity and mask fill values using
     * bit-wise or (|), see #FloodFillFlags.
     *
     * <b>Note:</b> Since the mask is larger than the filled image, a pixel \((x, y)\) in image corresponds to the
     * pixel \((x+1, y+1)\) in the mask .
     *
     * SEE: findContours
     * @return automatically generated
     */
    public static int floodFill(Mat image, Mat mask, Point seedPoint, Scalar newVal, Rect rect, Scalar loDiff, Scalar upDiff) {
        double[] rect_out = new double[4];
        int retVal = floodFill_1(image.nativeObj, mask.nativeObj, seedPoint.x, seedPoint.y, newVal.val[0], newVal.val[1], newVal.val[2], newVal.val[3], rect_out, loDiff.val[0], loDiff.val[1], loDiff.val[2], loDiff.val[3], upDiff.val[0], upDiff.val[1], upDiff.val[2], upDiff.val[3]);
        if(rect!=null){ rect.x = (int)rect_out[0]; rect.y = (int)rect_out[1]; rect.width = (int)rect_out[2]; rect.height = (int)rect_out[3]; } 
        return retVal;
    }

    /**
     * Fills a connected component with the given color.
     *
     * The function cv::floodFill fills a connected component starting from the seed point with the specified
     * color. The connectivity is determined by the color/brightness closeness of the neighbor pixels. The
     * pixel at \((x,y)\) is considered to belong to the repainted domain if:
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and floating range
     * \(\texttt{src} (x',y')- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} (x',y')+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and floating range
     * \(\texttt{src} (x',y')_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} (x',y')_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} (x',y')_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} (x',y')_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} (x',y')_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} (x',y')_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * where \(src(x',y')\) is the value of one of pixel neighbors that is already known to belong to the
     * component. That is, to be added to the connected component, a color/brightness of the pixel should
     * be close enough to:
     * <ul>
     *   <li>
     *  Color/brightness of one of its neighbors that already belong to the connected component in case
     * of a floating range.
     *   </li>
     *   <li>
     *  Color/brightness of the seed point in case of a fixed range.
     *   </li>
     * </ul>
     *
     * Use these functions to either mark a connected component with the specified color in-place, or build
     * a mask and then extract the contour, or copy the region to another image, and so on.
     *
     * @param image Input/output 1- or 3-channel, 8-bit, or floating-point image. It is modified by the
     * function unless the #FLOODFILL_MASK_ONLY flag is set in the second variant of the function. See
     * the details below.
     * @param mask Operation mask that should be a single-channel 8-bit image, 2 pixels wider and 2 pixels
     * taller than image. Since this is both an input and output parameter, you must take responsibility
     * of initializing it. Flood-filling cannot go across non-zero pixels in the input mask. For example,
     * an edge detector output can be used as a mask to stop filling at edges. On output, pixels in the
     * mask corresponding to filled pixels in the image are set to 1 or to the a value specified in flags
     * as described below. Additionally, the function fills the border of the mask with ones to simplify
     * internal processing. It is therefore possible to use the same mask in multiple calls to the function
     * to make sure the filled areas do not overlap.
     * @param seedPoint Starting point.
     * @param newVal New value of the repainted domain pixels.
     * @param loDiff Maximal lower brightness/color difference between the currently observed pixel and
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * @param rect Optional output parameter set by the function to the minimum bounding rectangle of the
     * repainted domain.
     * 4 means that only the four nearest neighbor pixels (those that share an edge) are considered. A
     * connectivity value of 8 means that the eight nearest neighbor pixels (those that share a corner)
     * will be considered. The next 8 bits (8-16) contain a value between 1 and 255 with which to fill
     * the mask (the default value is 1). For example, 4 | ( 255 &lt;&lt; 8 ) will consider 4 nearest
     * neighbours and fill the mask with a value of 255. The following additional options occupy higher
     * bits and therefore may be further combined with the connectivity and mask fill values using
     * bit-wise or (|), see #FloodFillFlags.
     *
     * <b>Note:</b> Since the mask is larger than the filled image, a pixel \((x, y)\) in image corresponds to the
     * pixel \((x+1, y+1)\) in the mask .
     *
     * SEE: findContours
     * @return automatically generated
     */
    public static int floodFill(Mat image, Mat mask, Point seedPoint, Scalar newVal, Rect rect, Scalar loDiff) {
        double[] rect_out = new double[4];
        int retVal = floodFill_2(image.nativeObj, mask.nativeObj, seedPoint.x, seedPoint.y, newVal.val[0], newVal.val[1], newVal.val[2], newVal.val[3], rect_out, loDiff.val[0], loDiff.val[1], loDiff.val[2], loDiff.val[3]);
        if(rect!=null){ rect.x = (int)rect_out[0]; rect.y = (int)rect_out[1]; rect.width = (int)rect_out[2]; rect.height = (int)rect_out[3]; } 
        return retVal;
    }

    /**
     * Fills a connected component with the given color.
     *
     * The function cv::floodFill fills a connected component starting from the seed point with the specified
     * color. The connectivity is determined by the color/brightness closeness of the neighbor pixels. The
     * pixel at \((x,y)\) is considered to belong to the repainted domain if:
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and floating range
     * \(\texttt{src} (x',y')- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} (x',y')+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and floating range
     * \(\texttt{src} (x',y')_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} (x',y')_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} (x',y')_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} (x',y')_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} (x',y')_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} (x',y')_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * where \(src(x',y')\) is the value of one of pixel neighbors that is already known to belong to the
     * component. That is, to be added to the connected component, a color/brightness of the pixel should
     * be close enough to:
     * <ul>
     *   <li>
     *  Color/brightness of one of its neighbors that already belong to the connected component in case
     * of a floating range.
     *   </li>
     *   <li>
     *  Color/brightness of the seed point in case of a fixed range.
     *   </li>
     * </ul>
     *
     * Use these functions to either mark a connected component with the specified color in-place, or build
     * a mask and then extract the contour, or copy the region to another image, and so on.
     *
     * @param image Input/output 1- or 3-channel, 8-bit, or floating-point image. It is modified by the
     * function unless the #FLOODFILL_MASK_ONLY flag is set in the second variant of the function. See
     * the details below.
     * @param mask Operation mask that should be a single-channel 8-bit image, 2 pixels wider and 2 pixels
     * taller than image. Since this is both an input and output parameter, you must take responsibility
     * of initializing it. Flood-filling cannot go across non-zero pixels in the input mask. For example,
     * an edge detector output can be used as a mask to stop filling at edges. On output, pixels in the
     * mask corresponding to filled pixels in the image are set to 1 or to the a value specified in flags
     * as described below. Additionally, the function fills the border of the mask with ones to simplify
     * internal processing. It is therefore possible to use the same mask in multiple calls to the function
     * to make sure the filled areas do not overlap.
     * @param seedPoint Starting point.
     * @param newVal New value of the repainted domain pixels.
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * @param rect Optional output parameter set by the function to the minimum bounding rectangle of the
     * repainted domain.
     * 4 means that only the four nearest neighbor pixels (those that share an edge) are considered. A
     * connectivity value of 8 means that the eight nearest neighbor pixels (those that share a corner)
     * will be considered. The next 8 bits (8-16) contain a value between 1 and 255 with which to fill
     * the mask (the default value is 1). For example, 4 | ( 255 &lt;&lt; 8 ) will consider 4 nearest
     * neighbours and fill the mask with a value of 255. The following additional options occupy higher
     * bits and therefore may be further combined with the connectivity and mask fill values using
     * bit-wise or (|), see #FloodFillFlags.
     *
     * <b>Note:</b> Since the mask is larger than the filled image, a pixel \((x, y)\) in image corresponds to the
     * pixel \((x+1, y+1)\) in the mask .
     *
     * SEE: findContours
     * @return automatically generated
     */
    public static int floodFill(Mat image, Mat mask, Point seedPoint, Scalar newVal, Rect rect) {
        double[] rect_out = new double[4];
        int retVal = floodFill_3(image.nativeObj, mask.nativeObj, seedPoint.x, seedPoint.y, newVal.val[0], newVal.val[1], newVal.val[2], newVal.val[3], rect_out);
        if(rect!=null){ rect.x = (int)rect_out[0]; rect.y = (int)rect_out[1]; rect.width = (int)rect_out[2]; rect.height = (int)rect_out[3]; } 
        return retVal;
    }

    /**
     * Fills a connected component with the given color.
     *
     * The function cv::floodFill fills a connected component starting from the seed point with the specified
     * color. The connectivity is determined by the color/brightness closeness of the neighbor pixels. The
     * pixel at \((x,y)\) is considered to belong to the repainted domain if:
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and floating range
     * \(\texttt{src} (x',y')- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} (x',y')+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a grayscale image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)+ \texttt{upDiff}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and floating range
     * \(\texttt{src} (x',y')_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} (x',y')_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} (x',y')_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} (x',y')_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} (x',y')_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} (x',y')_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  in case of a color image and fixed range
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r+ \texttt{upDiff} _r,\)
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g+ \texttt{upDiff} _g\)
     * and
     * \(\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b+ \texttt{upDiff} _b\)
     *   </li>
     * </ul>
     *
     *
     * where \(src(x',y')\) is the value of one of pixel neighbors that is already known to belong to the
     * component. That is, to be added to the connected component, a color/brightness of the pixel should
     * be close enough to:
     * <ul>
     *   <li>
     *  Color/brightness of one of its neighbors that already belong to the connected component in case
     * of a floating range.
     *   </li>
     *   <li>
     *  Color/brightness of the seed point in case of a fixed range.
     *   </li>
     * </ul>
     *
     * Use these functions to either mark a connected component with the specified color in-place, or build
     * a mask and then extract the contour, or copy the region to another image, and so on.
     *
     * @param image Input/output 1- or 3-channel, 8-bit, or floating-point image. It is modified by the
     * function unless the #FLOODFILL_MASK_ONLY flag is set in the second variant of the function. See
     * the details below.
     * @param mask Operation mask that should be a single-channel 8-bit image, 2 pixels wider and 2 pixels
     * taller than image. Since this is both an input and output parameter, you must take responsibility
     * of initializing it. Flood-filling cannot go across non-zero pixels in the input mask. For example,
     * an edge detector output can be used as a mask to stop filling at edges. On output, pixels in the
     * mask corresponding to filled pixels in the image are set to 1 or to the a value specified in flags
     * as described below. Additionally, the function fills the border of the mask with ones to simplify
     * internal processing. It is therefore possible to use the same mask in multiple calls to the function
     * to make sure the filled areas do not overlap.
     * @param seedPoint Starting point.
     * @param newVal New value of the repainted domain pixels.
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * one of its neighbors belonging to the component, or a seed pixel being added to the component.
     * repainted domain.
     * 4 means that only the four nearest neighbor pixels (those that share an edge) are considered. A
     * connectivity value of 8 means that the eight nearest neighbor pixels (those that share a corner)
     * will be considered. The next 8 bits (8-16) contain a value between 1 and 255 with which to fill
     * the mask (the default value is 1). For example, 4 | ( 255 &lt;&lt; 8 ) will consider 4 nearest
     * neighbours and fill the mask with a value of 255. The following additional options occupy higher
     * bits and therefore may be further combined with the connectivity and mask fill values using
     * bit-wise or (|), see #FloodFillFlags.
     *
     * <b>Note:</b> Since the mask is larger than the filled image, a pixel \((x, y)\) in image corresponds to the
     * pixel \((x+1, y+1)\) in the mask .
     *
     * SEE: findContours
     * @return automatically generated
     */
    public static int floodFill(Mat image, Mat mask, Point seedPoint, Scalar newVal) {
        return floodFill_4(image.nativeObj, mask.nativeObj, seedPoint.x, seedPoint.y, newVal.val[0], newVal.val[1], newVal.val[2], newVal.val[3]);
    }


    //
    // C++:  int cv::rotatedRectangleIntersection(RotatedRect rect1, RotatedRect rect2, Mat& intersectingRegion)
    //

    /**
     * Finds out if there is any intersection between two rotated rectangles.
     *
     * If there is then the vertices of the intersecting region are returned as well.
     *
     * Below are some examples of intersection configurations. The hatched pattern indicates the
     * intersecting region and the red vertices are returned by the function.
     *
     * ![intersection examples](pics/intersection.png)
     *
     * @param rect1 First rectangle
     * @param rect2 Second rectangle
     * @param intersectingRegion The output array of the vertices of the intersecting region. It returns
     * at most 8 vertices. Stored as std::vector&lt;cv::Point2f&gt; or cv::Mat as Mx1 of type CV_32FC2.
     * @return One of #RectanglesIntersectTypes
     */
    public static int rotatedRectangleIntersection(RotatedRect rect1, RotatedRect rect2, Mat intersectingRegion) {
        return rotatedRectangleIntersection_0(rect1.center.x, rect1.center.y, rect1.size.width, rect1.size.height, rect1.angle, rect2.center.x, rect2.center.y, rect2.size.width, rect2.size.height, rect2.angle, intersectingRegion.nativeObj);
    }


    //
    // C++:  void cv::Canny(Mat dx, Mat dy, Mat& edges, double threshold1, double threshold2, bool L2gradient = false)
    //

    /**
     * \overload
     *
     * Finds edges in an image using the Canny algorithm with custom image gradient.
     *
     * @param dx 16-bit x derivative of input image (CV_16SC1 or CV_16SC3).
     * @param dy 16-bit y derivative of input image (same type as dx).
     * @param edges output edge map; single channels 8-bit image, which has the same size as image .
     * @param threshold1 first threshold for the hysteresis procedure.
     * @param threshold2 second threshold for the hysteresis procedure.
     * @param L2gradient a flag, indicating whether a more accurate \(L_2\) norm
     * \(=\sqrt{(dI/dx)^2 + (dI/dy)^2}\) should be used to calculate the image gradient magnitude (
     * L2gradient=true ), or whether the default \(L_1\) norm \(=|dI/dx|+|dI/dy|\) is enough (
     * L2gradient=false ).
     */
    public static void Canny(Mat dx, Mat dy, Mat edges, double threshold1, double threshold2, boolean L2gradient) {
        Canny_0(dx.nativeObj, dy.nativeObj, edges.nativeObj, threshold1, threshold2, L2gradient);
    }

    /**
     * \overload
     *
     * Finds edges in an image using the Canny algorithm with custom image gradient.
     *
     * @param dx 16-bit x derivative of input image (CV_16SC1 or CV_16SC3).
     * @param dy 16-bit y derivative of input image (same type as dx).
     * @param edges output edge map; single channels 8-bit image, which has the same size as image .
     * @param threshold1 first threshold for the hysteresis procedure.
     * @param threshold2 second threshold for the hysteresis procedure.
     * \(=\sqrt{(dI/dx)^2 + (dI/dy)^2}\) should be used to calculate the image gradient magnitude (
     * L2gradient=true ), or whether the default \(L_1\) norm \(=|dI/dx|+|dI/dy|\) is enough (
     * L2gradient=false ).
     */
    public static void Canny(Mat dx, Mat dy, Mat edges, double threshold1, double threshold2) {
        Canny_1(dx.nativeObj, dy.nativeObj, edges.nativeObj, threshold1, threshold2);
    }


    //
    // C++:  void cv::Canny(Mat image, Mat& edges, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false)
    //

    /**
     * Finds edges in an image using the Canny algorithm CITE: Canny86 .
     *
     * The function finds edges in the input image and marks them in the output map edges using the
     * Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
     * largest value is used to find initial segments of strong edges. See
     * &lt;http://en.wikipedia.org/wiki/Canny_edge_detector&gt;
     *
     * @param image 8-bit input image.
     * @param edges output edge map; single channels 8-bit image, which has the same size as image .
     * @param threshold1 first threshold for the hysteresis procedure.
     * @param threshold2 second threshold for the hysteresis procedure.
     * @param apertureSize aperture size for the Sobel operator.
     * @param L2gradient a flag, indicating whether a more accurate \(L_2\) norm
     * \(=\sqrt{(dI/dx)^2 + (dI/dy)^2}\) should be used to calculate the image gradient magnitude (
     * L2gradient=true ), or whether the default \(L_1\) norm \(=|dI/dx|+|dI/dy|\) is enough (
     * L2gradient=false ).
     */
    public static void Canny(Mat image, Mat edges, double threshold1, double threshold2, int apertureSize, boolean L2gradient) {
        Canny_2(image.nativeObj, edges.nativeObj, threshold1, threshold2, apertureSize, L2gradient);
    }

    /**
     * Finds edges in an image using the Canny algorithm CITE: Canny86 .
     *
     * The function finds edges in the input image and marks them in the output map edges using the
     * Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
     * largest value is used to find initial segments of strong edges. See
     * &lt;http://en.wikipedia.org/wiki/Canny_edge_detector&gt;
     *
     * @param image 8-bit input image.
     * @param edges output edge map; single channels 8-bit image, which has the same size as image .
     * @param threshold1 first threshold for the hysteresis procedure.
     * @param threshold2 second threshold for the hysteresis procedure.
     * @param apertureSize aperture size for the Sobel operator.
     * \(=\sqrt{(dI/dx)^2 + (dI/dy)^2}\) should be used to calculate the image gradient magnitude (
     * L2gradient=true ), or whether the default \(L_1\) norm \(=|dI/dx|+|dI/dy|\) is enough (
     * L2gradient=false ).
     */
    public static void Canny(Mat image, Mat edges, double threshold1, double threshold2, int apertureSize) {
        Canny_3(image.nativeObj, edges.nativeObj, threshold1, threshold2, apertureSize);
    }

    /**
     * Finds edges in an image using the Canny algorithm CITE: Canny86 .
     *
     * The function finds edges in the input image and marks them in the output map edges using the
     * Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
     * largest value is used to find initial segments of strong edges. See
     * &lt;http://en.wikipedia.org/wiki/Canny_edge_detector&gt;
     *
     * @param image 8-bit input image.
     * @param edges output edge map; single channels 8-bit image, which has the same size as image .
     * @param threshold1 first threshold for the hysteresis procedure.
     * @param threshold2 second threshold for the hysteresis procedure.
     * \(=\sqrt{(dI/dx)^2 + (dI/dy)^2}\) should be used to calculate the image gradient magnitude (
     * L2gradient=true ), or whether the default \(L_1\) norm \(=|dI/dx|+|dI/dy|\) is enough (
     * L2gradient=false ).
     */
    public static void Canny(Mat image, Mat edges, double threshold1, double threshold2) {
        Canny_4(image.nativeObj, edges.nativeObj, threshold1, threshold2);
    }


    //
    // C++:  void cv::GaussianBlur(Mat src, Mat& dst, Size ksize, double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT)
    //

    /**
     * Blurs an image using a Gaussian filter.
     *
     * The function convolves the source image with the specified Gaussian kernel. In-place filtering is
     * supported.
     *
     * @param src input image; the image can have any number of channels, which are processed
     * independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
     * positive and odd. Or, they can be zero's and then they are computed from sigma.
     * @param sigmaX Gaussian kernel standard deviation in X direction.
     * @param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be
     * equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,
     * respectively (see #getGaussianKernel for details); to fully control the result regardless of
     * possible future modifications of all this semantics, it is recommended to specify all of ksize,
     * sigmaX, and sigmaY.
     * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     *
     * SEE:  sepFilter2D, filter2D, blur, boxFilter, bilateralFilter, medianBlur
     */
    public static void GaussianBlur(Mat src, Mat dst, Size ksize, double sigmaX, double sigmaY, int borderType) {
        GaussianBlur_0(src.nativeObj, dst.nativeObj, ksize.width, ksize.height, sigmaX, sigmaY, borderType);
    }

    /**
     * Blurs an image using a Gaussian filter.
     *
     * The function convolves the source image with the specified Gaussian kernel. In-place filtering is
     * supported.
     *
     * @param src input image; the image can have any number of channels, which are processed
     * independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
     * positive and odd. Or, they can be zero's and then they are computed from sigma.
     * @param sigmaX Gaussian kernel standard deviation in X direction.
     * @param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be
     * equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,
     * respectively (see #getGaussianKernel for details); to fully control the result regardless of
     * possible future modifications of all this semantics, it is recommended to specify all of ksize,
     * sigmaX, and sigmaY.
     *
     * SEE:  sepFilter2D, filter2D, blur, boxFilter, bilateralFilter, medianBlur
     */
    public static void GaussianBlur(Mat src, Mat dst, Size ksize, double sigmaX, double sigmaY) {
        GaussianBlur_1(src.nativeObj, dst.nativeObj, ksize.width, ksize.height, sigmaX, sigmaY);
    }

    /**
     * Blurs an image using a Gaussian filter.
     *
     * The function convolves the source image with the specified Gaussian kernel. In-place filtering is
     * supported.
     *
     * @param src input image; the image can have any number of channels, which are processed
     * independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
     * positive and odd. Or, they can be zero's and then they are computed from sigma.
     * @param sigmaX Gaussian kernel standard deviation in X direction.
     * equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,
     * respectively (see #getGaussianKernel for details); to fully control the result regardless of
     * possible future modifications of all this semantics, it is recommended to specify all of ksize,
     * sigmaX, and sigmaY.
     *
     * SEE:  sepFilter2D, filter2D, blur, boxFilter, bilateralFilter, medianBlur
     */
    public static void GaussianBlur(Mat src, Mat dst, Size ksize, double sigmaX) {
        GaussianBlur_2(src.nativeObj, dst.nativeObj, ksize.width, ksize.height, sigmaX);
    }


    //
    // C++:  void cv::HoughCircles(Mat image, Mat& circles, int method, double dp, double minDist, double param1 = 100, double param2 = 100, int minRadius = 0, int maxRadius = 0)
    //

    /**
     * Finds circles in a grayscale image using the Hough transform.
     *
     * The function finds circles in a grayscale image using a modification of the Hough transform.
     *
     * Example: :
     * INCLUDE: snippets/imgproc_HoughLinesCircles.cpp
     *
     * <b>Note:</b> Usually the function detects the centers of circles well. However, it may fail to find correct
     * radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
     * you know it. Or, in the case of #HOUGH_GRADIENT method you may set maxRadius to a negative number
     * to return centers only without radius search, and find the correct radius using an additional procedure.
     *
     * It also helps to smooth image a bit unless it's already soft. For example,
     * GaussianBlur() with 7x7 kernel and 1.5x1.5 sigma or similar blurring may help.
     *
     * @param image 8-bit, single-channel, grayscale input image.
     * @param circles Output vector of found circles. Each vector is encoded as  3 or 4 element
     * floating-point vector \((x, y, radius)\) or \((x, y, radius, votes)\) .
     * @param method Detection method, see #HoughModes. The available methods are #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT.
     * @param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if
     * dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has
     * half as big width and height. For #HOUGH_GRADIENT_ALT the recommended value is dp=1.5,
     * unless some small very circles need to be detected.
     * @param minDist Minimum distance between the centers of the detected circles. If the parameter is
     * too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is
     * too large, some circles may be missed.
     * @param param1 First method-specific parameter. In case of #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT,
     * it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
     * Note that #HOUGH_GRADIENT_ALT uses #Scharr algorithm to compute image derivatives, so the threshold value
     * shough normally be higher, such as 300 or normally exposed and contrasty images.
     * @param param2 Second method-specific parameter. In case of #HOUGH_GRADIENT, it is the
     * accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
     * false circles may be detected. Circles, corresponding to the larger accumulator values, will be
     * returned first. In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure.
     * The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine.
     * If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less.
     * But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
     * @param minRadius Minimum circle radius.
     * @param maxRadius Maximum circle radius. If &lt;= 0, uses the maximum image dimension. If &lt; 0, #HOUGH_GRADIENT returns
     * centers without finding the radius. #HOUGH_GRADIENT_ALT always computes circle radiuses.
     *
     * SEE: fitEllipse, minEnclosingCircle
     */
    public static void HoughCircles(Mat image, Mat circles, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius) {
        HoughCircles_0(image.nativeObj, circles.nativeObj, method, dp, minDist, param1, param2, minRadius, maxRadius);
    }

    /**
     * Finds circles in a grayscale image using the Hough transform.
     *
     * The function finds circles in a grayscale image using a modification of the Hough transform.
     *
     * Example: :
     * INCLUDE: snippets/imgproc_HoughLinesCircles.cpp
     *
     * <b>Note:</b> Usually the function detects the centers of circles well. However, it may fail to find correct
     * radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
     * you know it. Or, in the case of #HOUGH_GRADIENT method you may set maxRadius to a negative number
     * to return centers only without radius search, and find the correct radius using an additional procedure.
     *
     * It also helps to smooth image a bit unless it's already soft. For example,
     * GaussianBlur() with 7x7 kernel and 1.5x1.5 sigma or similar blurring may help.
     *
     * @param image 8-bit, single-channel, grayscale input image.
     * @param circles Output vector of found circles. Each vector is encoded as  3 or 4 element
     * floating-point vector \((x, y, radius)\) or \((x, y, radius, votes)\) .
     * @param method Detection method, see #HoughModes. The available methods are #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT.
     * @param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if
     * dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has
     * half as big width and height. For #HOUGH_GRADIENT_ALT the recommended value is dp=1.5,
     * unless some small very circles need to be detected.
     * @param minDist Minimum distance between the centers of the detected circles. If the parameter is
     * too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is
     * too large, some circles may be missed.
     * @param param1 First method-specific parameter. In case of #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT,
     * it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
     * Note that #HOUGH_GRADIENT_ALT uses #Scharr algorithm to compute image derivatives, so the threshold value
     * shough normally be higher, such as 300 or normally exposed and contrasty images.
     * @param param2 Second method-specific parameter. In case of #HOUGH_GRADIENT, it is the
     * accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
     * false circles may be detected. Circles, corresponding to the larger accumulator values, will be
     * returned first. In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure.
     * The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine.
     * If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less.
     * But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
     * @param minRadius Minimum circle radius.
     * centers without finding the radius. #HOUGH_GRADIENT_ALT always computes circle radiuses.
     *
     * SEE: fitEllipse, minEnclosingCircle
     */
    public static void HoughCircles(Mat image, Mat circles, int method, double dp, double minDist, double param1, double param2, int minRadius) {
        HoughCircles_1(image.nativeObj, circles.nativeObj, method, dp, minDist, param1, param2, minRadius);
    }

    /**
     * Finds circles in a grayscale image using the Hough transform.
     *
     * The function finds circles in a grayscale image using a modification of the Hough transform.
     *
     * Example: :
     * INCLUDE: snippets/imgproc_HoughLinesCircles.cpp
     *
     * <b>Note:</b> Usually the function detects the centers of circles well. However, it may fail to find correct
     * radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
     * you know it. Or, in the case of #HOUGH_GRADIENT method you may set maxRadius to a negative number
     * to return centers only without radius search, and find the correct radius using an additional procedure.
     *
     * It also helps to smooth image a bit unless it's already soft. For example,
     * GaussianBlur() with 7x7 kernel and 1.5x1.5 sigma or similar blurring may help.
     *
     * @param image 8-bit, single-channel, grayscale input image.
     * @param circles Output vector of found circles. Each vector is encoded as  3 or 4 element
     * floating-point vector \((x, y, radius)\) or \((x, y, radius, votes)\) .
     * @param method Detection method, see #HoughModes. The available methods are #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT.
     * @param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if
     * dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has
     * half as big width and height. For #HOUGH_GRADIENT_ALT the recommended value is dp=1.5,
     * unless some small very circles need to be detected.
     * @param minDist Minimum distance between the centers of the detected circles. If the parameter is
     * too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is
     * too large, some circles may be missed.
     * @param param1 First method-specific parameter. In case of #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT,
     * it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
     * Note that #HOUGH_GRADIENT_ALT uses #Scharr algorithm to compute image derivatives, so the threshold value
     * shough normally be higher, such as 300 or normally exposed and contrasty images.
     * @param param2 Second method-specific parameter. In case of #HOUGH_GRADIENT, it is the
     * accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
     * false circles may be detected. Circles, corresponding to the larger accumulator values, will be
     * returned first. In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure.
     * The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine.
     * If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less.
     * But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
     * centers without finding the radius. #HOUGH_GRADIENT_ALT always computes circle radiuses.
     *
     * SEE: fitEllipse, minEnclosingCircle
     */
    public static void HoughCircles(Mat image, Mat circles, int method, double dp, double minDist, double param1, double param2) {
        HoughCircles_2(image.nativeObj, circles.nativeObj, method, dp, minDist, param1, param2);
    }

    /**
     * Finds circles in a grayscale image using the Hough transform.
     *
     * The function finds circles in a grayscale image using a modification of the Hough transform.
     *
     * Example: :
     * INCLUDE: snippets/imgproc_HoughLinesCircles.cpp
     *
     * <b>Note:</b> Usually the function detects the centers of circles well. However, it may fail to find correct
     * radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
     * you know it. Or, in the case of #HOUGH_GRADIENT method you may set maxRadius to a negative number
     * to return centers only without radius search, and find the correct radius using an additional procedure.
     *
     * It also helps to smooth image a bit unless it's already soft. For example,
     * GaussianBlur() with 7x7 kernel and 1.5x1.5 sigma or similar blurring may help.
     *
     * @param image 8-bit, single-channel, grayscale input image.
     * @param circles Output vector of found circles. Each vector is encoded as  3 or 4 element
     * floating-point vector \((x, y, radius)\) or \((x, y, radius, votes)\) .
     * @param method Detection method, see #HoughModes. The available methods are #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT.
     * @param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if
     * dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has
     * half as big width and height. For #HOUGH_GRADIENT_ALT the recommended value is dp=1.5,
     * unless some small very circles need to be detected.
     * @param minDist Minimum distance between the centers of the detected circles. If the parameter is
     * too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is
     * too large, some circles may be missed.
     * @param param1 First method-specific parameter. In case of #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT,
     * it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
     * Note that #HOUGH_GRADIENT_ALT uses #Scharr algorithm to compute image derivatives, so the threshold value
     * shough normally be higher, such as 300 or normally exposed and contrasty images.
     * accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
     * false circles may be detected. Circles, corresponding to the larger accumulator values, will be
     * returned first. In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure.
     * The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine.
     * If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less.
     * But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
     * centers without finding the radius. #HOUGH_GRADIENT_ALT always computes circle radiuses.
     *
     * SEE: fitEllipse, minEnclosingCircle
     */
    public static void HoughCircles(Mat image, Mat circles, int method, double dp, double minDist, double param1) {
        HoughCircles_3(image.nativeObj, circles.nativeObj, method, dp, minDist, param1);
    }

    /**
     * Finds circles in a grayscale image using the Hough transform.
     *
     * The function finds circles in a grayscale image using a modification of the Hough transform.
     *
     * Example: :
     * INCLUDE: snippets/imgproc_HoughLinesCircles.cpp
     *
     * <b>Note:</b> Usually the function detects the centers of circles well. However, it may fail to find correct
     * radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
     * you know it. Or, in the case of #HOUGH_GRADIENT method you may set maxRadius to a negative number
     * to return centers only without radius search, and find the correct radius using an additional procedure.
     *
     * It also helps to smooth image a bit unless it's already soft. For example,
     * GaussianBlur() with 7x7 kernel and 1.5x1.5 sigma or similar blurring may help.
     *
     * @param image 8-bit, single-channel, grayscale input image.
     * @param circles Output vector of found circles. Each vector is encoded as  3 or 4 element
     * floating-point vector \((x, y, radius)\) or \((x, y, radius, votes)\) .
     * @param method Detection method, see #HoughModes. The available methods are #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT.
     * @param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if
     * dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has
     * half as big width and height. For #HOUGH_GRADIENT_ALT the recommended value is dp=1.5,
     * unless some small very circles need to be detected.
     * @param minDist Minimum distance between the centers of the detected circles. If the parameter is
     * too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is
     * too large, some circles may be missed.
     * it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
     * Note that #HOUGH_GRADIENT_ALT uses #Scharr algorithm to compute image derivatives, so the threshold value
     * shough normally be higher, such as 300 or normally exposed and contrasty images.
     * accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
     * false circles may be detected. Circles, corresponding to the larger accumulator values, will be
     * returned first. In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure.
     * The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine.
     * If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less.
     * But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
     * centers without finding the radius. #HOUGH_GRADIENT_ALT always computes circle radiuses.
     *
     * SEE: fitEllipse, minEnclosingCircle
     */
    public static void HoughCircles(Mat image, Mat circles, int method, double dp, double minDist) {
        HoughCircles_4(image.nativeObj, circles.nativeObj, method, dp, minDist);
    }


    //
    // C++:  void cv::HoughLines(Mat image, Mat& lines, double rho, double theta, int threshold, double srn = 0, double stn = 0, double min_theta = 0, double max_theta = CV_PI)
    //

    /**
     * Finds lines in a binary image using the standard Hough transform.
     *
     * The function implements the standard or standard multi-scale Hough transform algorithm for line
     * detection. See &lt;http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm&gt; for a good explanation of Hough
     * transform.
     *
     * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines Output vector of lines. Each line is represented by a 2 or 3 element vector
     * \((\rho, \theta)\) or \((\rho, \theta, \textrm{votes})\) . \(\rho\) is the distance from the coordinate origin \((0,0)\) (top-left corner of
     * the image). \(\theta\) is the line rotation angle in radians (
     * \(0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}\) ).
     * \(\textrm{votes}\) is the value of accumulator.
     * @param rho Distance resolution of the accumulator in pixels.
     * @param theta Angle resolution of the accumulator in radians.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) ).
     * @param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho .
     * The coarse accumulator distance resolution is rho and the accurate accumulator resolution is
     * rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these
     * parameters should be positive.
     * @param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
     * @param min_theta For standard and multi-scale Hough transform, minimum angle to check for lines.
     * Must fall between 0 and max_theta.
     * @param max_theta For standard and multi-scale Hough transform, maximum angle to check for lines.
     * Must fall between min_theta and CV_PI.
     */
    public static void HoughLines(Mat image, Mat lines, double rho, double theta, int threshold, double srn, double stn, double min_theta, double max_theta) {
        HoughLines_0(image.nativeObj, lines.nativeObj, rho, theta, threshold, srn, stn, min_theta, max_theta);
    }

    /**
     * Finds lines in a binary image using the standard Hough transform.
     *
     * The function implements the standard or standard multi-scale Hough transform algorithm for line
     * detection. See &lt;http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm&gt; for a good explanation of Hough
     * transform.
     *
     * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines Output vector of lines. Each line is represented by a 2 or 3 element vector
     * \((\rho, \theta)\) or \((\rho, \theta, \textrm{votes})\) . \(\rho\) is the distance from the coordinate origin \((0,0)\) (top-left corner of
     * the image). \(\theta\) is the line rotation angle in radians (
     * \(0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}\) ).
     * \(\textrm{votes}\) is the value of accumulator.
     * @param rho Distance resolution of the accumulator in pixels.
     * @param theta Angle resolution of the accumulator in radians.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) ).
     * @param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho .
     * The coarse accumulator distance resolution is rho and the accurate accumulator resolution is
     * rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these
     * parameters should be positive.
     * @param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
     * @param min_theta For standard and multi-scale Hough transform, minimum angle to check for lines.
     * Must fall between 0 and max_theta.
     * Must fall between min_theta and CV_PI.
     */
    public static void HoughLines(Mat image, Mat lines, double rho, double theta, int threshold, double srn, double stn, double min_theta) {
        HoughLines_1(image.nativeObj, lines.nativeObj, rho, theta, threshold, srn, stn, min_theta);
    }

    /**
     * Finds lines in a binary image using the standard Hough transform.
     *
     * The function implements the standard or standard multi-scale Hough transform algorithm for line
     * detection. See &lt;http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm&gt; for a good explanation of Hough
     * transform.
     *
     * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines Output vector of lines. Each line is represented by a 2 or 3 element vector
     * \((\rho, \theta)\) or \((\rho, \theta, \textrm{votes})\) . \(\rho\) is the distance from the coordinate origin \((0,0)\) (top-left corner of
     * the image). \(\theta\) is the line rotation angle in radians (
     * \(0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}\) ).
     * \(\textrm{votes}\) is the value of accumulator.
     * @param rho Distance resolution of the accumulator in pixels.
     * @param theta Angle resolution of the accumulator in radians.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) ).
     * @param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho .
     * The coarse accumulator distance resolution is rho and the accurate accumulator resolution is
     * rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these
     * parameters should be positive.
     * @param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
     * Must fall between 0 and max_theta.
     * Must fall between min_theta and CV_PI.
     */
    public static void HoughLines(Mat image, Mat lines, double rho, double theta, int threshold, double srn, double stn) {
        HoughLines_2(image.nativeObj, lines.nativeObj, rho, theta, threshold, srn, stn);
    }

    /**
     * Finds lines in a binary image using the standard Hough transform.
     *
     * The function implements the standard or standard multi-scale Hough transform algorithm for line
     * detection. See &lt;http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm&gt; for a good explanation of Hough
     * transform.
     *
     * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines Output vector of lines. Each line is represented by a 2 or 3 element vector
     * \((\rho, \theta)\) or \((\rho, \theta, \textrm{votes})\) . \(\rho\) is the distance from the coordinate origin \((0,0)\) (top-left corner of
     * the image). \(\theta\) is the line rotation angle in radians (
     * \(0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}\) ).
     * \(\textrm{votes}\) is the value of accumulator.
     * @param rho Distance resolution of the accumulator in pixels.
     * @param theta Angle resolution of the accumulator in radians.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) ).
     * @param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho .
     * The coarse accumulator distance resolution is rho and the accurate accumulator resolution is
     * rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these
     * parameters should be positive.
     * Must fall between 0 and max_theta.
     * Must fall between min_theta and CV_PI.
     */
    public static void HoughLines(Mat image, Mat lines, double rho, double theta, int threshold, double srn) {
        HoughLines_3(image.nativeObj, lines.nativeObj, rho, theta, threshold, srn);
    }

    /**
     * Finds lines in a binary image using the standard Hough transform.
     *
     * The function implements the standard or standard multi-scale Hough transform algorithm for line
     * detection. See &lt;http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm&gt; for a good explanation of Hough
     * transform.
     *
     * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines Output vector of lines. Each line is represented by a 2 or 3 element vector
     * \((\rho, \theta)\) or \((\rho, \theta, \textrm{votes})\) . \(\rho\) is the distance from the coordinate origin \((0,0)\) (top-left corner of
     * the image). \(\theta\) is the line rotation angle in radians (
     * \(0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}\) ).
     * \(\textrm{votes}\) is the value of accumulator.
     * @param rho Distance resolution of the accumulator in pixels.
     * @param theta Angle resolution of the accumulator in radians.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) ).
     * The coarse accumulator distance resolution is rho and the accurate accumulator resolution is
     * rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these
     * parameters should be positive.
     * Must fall between 0 and max_theta.
     * Must fall between min_theta and CV_PI.
     */
    public static void HoughLines(Mat image, Mat lines, double rho, double theta, int threshold) {
        HoughLines_4(image.nativeObj, lines.nativeObj, rho, theta, threshold);
    }


    //
    // C++:  void cv::HoughLinesP(Mat image, Mat& lines, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0)
    //

    /**
     * Finds line segments in a binary image using the probabilistic Hough transform.
     *
     * The function implements the probabilistic Hough transform algorithm for line detection, described
     * in CITE: Matas00
     *
     * See the line detection example below:
     * INCLUDE: snippets/imgproc_HoughLinesP.cpp
     * This is a sample picture the function parameters have been tuned for:
     *
     * ![image](pics/building.jpg)
     *
     * And this is the output of the above program in case of the probabilistic Hough transform:
     *
     * ![image](pics/houghp.png)
     *
     * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines Output vector of lines. Each line is represented by a 4-element vector
     * \((x_1, y_1, x_2, y_2)\) , where \((x_1,y_1)\) and \((x_2, y_2)\) are the ending points of each detected
     * line segment.
     * @param rho Distance resolution of the accumulator in pixels.
     * @param theta Angle resolution of the accumulator in radians.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) ).
     * @param minLineLength Minimum line length. Line segments shorter than that are rejected.
     * @param maxLineGap Maximum allowed gap between points on the same line to link them.
     *
     * SEE: LineSegmentDetector
     */
    public static void HoughLinesP(Mat image, Mat lines, double rho, double theta, int threshold, double minLineLength, double maxLineGap) {
        HoughLinesP_0(image.nativeObj, lines.nativeObj, rho, theta, threshold, minLineLength, maxLineGap);
    }

    /**
     * Finds line segments in a binary image using the probabilistic Hough transform.
     *
     * The function implements the probabilistic Hough transform algorithm for line detection, described
     * in CITE: Matas00
     *
     * See the line detection example below:
     * INCLUDE: snippets/imgproc_HoughLinesP.cpp
     * This is a sample picture the function parameters have been tuned for:
     *
     * ![image](pics/building.jpg)
     *
     * And this is the output of the above program in case of the probabilistic Hough transform:
     *
     * ![image](pics/houghp.png)
     *
     * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines Output vector of lines. Each line is represented by a 4-element vector
     * \((x_1, y_1, x_2, y_2)\) , where \((x_1,y_1)\) and \((x_2, y_2)\) are the ending points of each detected
     * line segment.
     * @param rho Distance resolution of the accumulator in pixels.
     * @param theta Angle resolution of the accumulator in radians.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) ).
     * @param minLineLength Minimum line length. Line segments shorter than that are rejected.
     *
     * SEE: LineSegmentDetector
     */
    public static void HoughLinesP(Mat image, Mat lines, double rho, double theta, int threshold, double minLineLength) {
        HoughLinesP_1(image.nativeObj, lines.nativeObj, rho, theta, threshold, minLineLength);
    }

    /**
     * Finds line segments in a binary image using the probabilistic Hough transform.
     *
     * The function implements the probabilistic Hough transform algorithm for line detection, described
     * in CITE: Matas00
     *
     * See the line detection example below:
     * INCLUDE: snippets/imgproc_HoughLinesP.cpp
     * This is a sample picture the function parameters have been tuned for:
     *
     * ![image](pics/building.jpg)
     *
     * And this is the output of the above program in case of the probabilistic Hough transform:
     *
     * ![image](pics/houghp.png)
     *
     * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines Output vector of lines. Each line is represented by a 4-element vector
     * \((x_1, y_1, x_2, y_2)\) , where \((x_1,y_1)\) and \((x_2, y_2)\) are the ending points of each detected
     * line segment.
     * @param rho Distance resolution of the accumulator in pixels.
     * @param theta Angle resolution of the accumulator in radians.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) ).
     *
     * SEE: LineSegmentDetector
     */
    public static void HoughLinesP(Mat image, Mat lines, double rho, double theta, int threshold) {
        HoughLinesP_2(image.nativeObj, lines.nativeObj, rho, theta, threshold);
    }


    //
    // C++:  void cv::HoughLinesPointSet(Mat _point, Mat& _lines, int lines_max, int threshold, double min_rho, double max_rho, double rho_step, double min_theta, double max_theta, double theta_step)
    //

    /**
     * Finds lines in a set of points using the standard Hough transform.
     *
     * The function finds lines in a set of points using a modification of the Hough transform.
     * INCLUDE: snippets/imgproc_HoughLinesPointSet.cpp
     * @param _point Input vector of points. Each vector must be encoded as a Point vector \((x,y)\). Type must be CV_32FC2 or CV_32SC2.
     * @param _lines Output vector of found lines. Each vector is encoded as a vector&lt;Vec3d&gt; \((votes, rho, theta)\).
     * The larger the value of 'votes', the higher the reliability of the Hough line.
     * @param lines_max Max count of hough lines.
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
     * votes ( \(&gt;\texttt{threshold}\) )
     * @param min_rho Minimum Distance value of the accumulator in pixels.
     * @param max_rho Maximum Distance value of the accumulator in pixels.
     * @param rho_step Distance resolution of the accumulator in pixels.
     * @param min_theta Minimum angle value of the accumulator in radians.
     * @param max_theta Maximum angle value of the accumulator in radians.
     * @param theta_step Angle resolution of the accumulator in radians.
     */
    public static void HoughLinesPointSet(Mat _point, Mat _lines, int lines_max, int threshold, double min_rho, double max_rho, double rho_step, double min_theta, double max_theta, double theta_step) {
        HoughLinesPointSet_0(_point.nativeObj, _lines.nativeObj, lines_max, threshold, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step);
    }


    //
    // C++:  void cv::HuMoments(Moments m, Mat& hu)
    //

    public static void HuMoments(Moments m, Mat hu) {
        HuMoments_0(m.m00, m.m10, m.m01, m.m20, m.m11, m.m02, m.m30, m.m21, m.m12, m.m03, hu.nativeObj);
    }


    //
    // C++:  void cv::Laplacian(Mat src, Mat& dst, int ddepth, int ksize = 1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
    //

    /**
     * Calculates the Laplacian of an image.
     *
     * The function calculates the Laplacian of the source image by adding up the second x and y
     * derivatives calculated using the Sobel operator:
     *
     * \(\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\)
     *
     * This is done when {@code ksize &gt; 1}. When {@code ksize == 1}, the Laplacian is computed by filtering the image
     * with the following \(3 \times 3\) aperture:
     *
     * \(\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\)
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Desired depth of the destination image.
     * @param ksize Aperture size used to compute the second-derivative filters. See #getDerivKernels for
     * details. The size must be positive and odd.
     * @param scale Optional scale factor for the computed Laplacian values. By default, no scaling is
     * applied. See #getDerivKernels for details.
     * @param delta Optional delta value that is added to the results prior to storing them in dst .
     * @param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE:  Sobel, Scharr
     */
    public static void Laplacian(Mat src, Mat dst, int ddepth, int ksize, double scale, double delta, int borderType) {
        Laplacian_0(src.nativeObj, dst.nativeObj, ddepth, ksize, scale, delta, borderType);
    }

    /**
     * Calculates the Laplacian of an image.
     *
     * The function calculates the Laplacian of the source image by adding up the second x and y
     * derivatives calculated using the Sobel operator:
     *
     * \(\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\)
     *
     * This is done when {@code ksize &gt; 1}. When {@code ksize == 1}, the Laplacian is computed by filtering the image
     * with the following \(3 \times 3\) aperture:
     *
     * \(\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\)
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Desired depth of the destination image.
     * @param ksize Aperture size used to compute the second-derivative filters. See #getDerivKernels for
     * details. The size must be positive and odd.
     * @param scale Optional scale factor for the computed Laplacian values. By default, no scaling is
     * applied. See #getDerivKernels for details.
     * @param delta Optional delta value that is added to the results prior to storing them in dst .
     * SEE:  Sobel, Scharr
     */
    public static void Laplacian(Mat src, Mat dst, int ddepth, int ksize, double scale, double delta) {
        Laplacian_1(src.nativeObj, dst.nativeObj, ddepth, ksize, scale, delta);
    }

    /**
     * Calculates the Laplacian of an image.
     *
     * The function calculates the Laplacian of the source image by adding up the second x and y
     * derivatives calculated using the Sobel operator:
     *
     * \(\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\)
     *
     * This is done when {@code ksize &gt; 1}. When {@code ksize == 1}, the Laplacian is computed by filtering the image
     * with the following \(3 \times 3\) aperture:
     *
     * \(\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\)
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Desired depth of the destination image.
     * @param ksize Aperture size used to compute the second-derivative filters. See #getDerivKernels for
     * details. The size must be positive and odd.
     * @param scale Optional scale factor for the computed Laplacian values. By default, no scaling is
     * applied. See #getDerivKernels for details.
     * SEE:  Sobel, Scharr
     */
    public static void Laplacian(Mat src, Mat dst, int ddepth, int ksize, double scale) {
        Laplacian_2(src.nativeObj, dst.nativeObj, ddepth, ksize, scale);
    }

    /**
     * Calculates the Laplacian of an image.
     *
     * The function calculates the Laplacian of the source image by adding up the second x and y
     * derivatives calculated using the Sobel operator:
     *
     * \(\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\)
     *
     * This is done when {@code ksize &gt; 1}. When {@code ksize == 1}, the Laplacian is computed by filtering the image
     * with the following \(3 \times 3\) aperture:
     *
     * \(\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\)
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Desired depth of the destination image.
     * @param ksize Aperture size used to compute the second-derivative filters. See #getDerivKernels for
     * details. The size must be positive and odd.
     * applied. See #getDerivKernels for details.
     * SEE:  Sobel, Scharr
     */
    public static void Laplacian(Mat src, Mat dst, int ddepth, int ksize) {
        Laplacian_3(src.nativeObj, dst.nativeObj, ddepth, ksize);
    }

    /**
     * Calculates the Laplacian of an image.
     *
     * The function calculates the Laplacian of the source image by adding up the second x and y
     * derivatives calculated using the Sobel operator:
     *
     * \(\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\)
     *
     * This is done when {@code ksize &gt; 1}. When {@code ksize == 1}, the Laplacian is computed by filtering the image
     * with the following \(3 \times 3\) aperture:
     *
     * \(\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\)
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Desired depth of the destination image.
     * details. The size must be positive and odd.
     * applied. See #getDerivKernels for details.
     * SEE:  Sobel, Scharr
     */
    public static void Laplacian(Mat src, Mat dst, int ddepth) {
        Laplacian_4(src.nativeObj, dst.nativeObj, ddepth);
    }


    //
    // C++:  void cv::Scharr(Mat src, Mat& dst, int ddepth, int dx, int dy, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
    //

    /**
     * Calculates the first x- or y- image derivative using Scharr operator.
     *
     * The function computes the first x- or y- spatial image derivative using the Scharr operator. The
     * call
     *
     * \(\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}\)
     *
     * is equivalent to
     *
     * \(\texttt{Sobel(src, dst, ddepth, dx, dy, FILTER_SCHARR, scale, delta, borderType)} .\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src.
     * @param ddepth output image depth, see REF: filter_depths "combinations"
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * @param scale optional scale factor for the computed derivative values; by default, no scaling is
     * applied (see #getDerivKernels for details).
     * @param delta optional delta value that is added to the results prior to storing them in dst.
     * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE:  cartToPolar
     */
    public static void Scharr(Mat src, Mat dst, int ddepth, int dx, int dy, double scale, double delta, int borderType) {
        Scharr_0(src.nativeObj, dst.nativeObj, ddepth, dx, dy, scale, delta, borderType);
    }

    /**
     * Calculates the first x- or y- image derivative using Scharr operator.
     *
     * The function computes the first x- or y- spatial image derivative using the Scharr operator. The
     * call
     *
     * \(\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}\)
     *
     * is equivalent to
     *
     * \(\texttt{Sobel(src, dst, ddepth, dx, dy, FILTER_SCHARR, scale, delta, borderType)} .\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src.
     * @param ddepth output image depth, see REF: filter_depths "combinations"
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * @param scale optional scale factor for the computed derivative values; by default, no scaling is
     * applied (see #getDerivKernels for details).
     * @param delta optional delta value that is added to the results prior to storing them in dst.
     * SEE:  cartToPolar
     */
    public static void Scharr(Mat src, Mat dst, int ddepth, int dx, int dy, double scale, double delta) {
        Scharr_1(src.nativeObj, dst.nativeObj, ddepth, dx, dy, scale, delta);
    }

    /**
     * Calculates the first x- or y- image derivative using Scharr operator.
     *
     * The function computes the first x- or y- spatial image derivative using the Scharr operator. The
     * call
     *
     * \(\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}\)
     *
     * is equivalent to
     *
     * \(\texttt{Sobel(src, dst, ddepth, dx, dy, FILTER_SCHARR, scale, delta, borderType)} .\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src.
     * @param ddepth output image depth, see REF: filter_depths "combinations"
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * @param scale optional scale factor for the computed derivative values; by default, no scaling is
     * applied (see #getDerivKernels for details).
     * SEE:  cartToPolar
     */
    public static void Scharr(Mat src, Mat dst, int ddepth, int dx, int dy, double scale) {
        Scharr_2(src.nativeObj, dst.nativeObj, ddepth, dx, dy, scale);
    }

    /**
     * Calculates the first x- or y- image derivative using Scharr operator.
     *
     * The function computes the first x- or y- spatial image derivative using the Scharr operator. The
     * call
     *
     * \(\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}\)
     *
     * is equivalent to
     *
     * \(\texttt{Sobel(src, dst, ddepth, dx, dy, FILTER_SCHARR, scale, delta, borderType)} .\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src.
     * @param ddepth output image depth, see REF: filter_depths "combinations"
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * applied (see #getDerivKernels for details).
     * SEE:  cartToPolar
     */
    public static void Scharr(Mat src, Mat dst, int ddepth, int dx, int dy) {
        Scharr_3(src.nativeObj, dst.nativeObj, ddepth, dx, dy);
    }


    //
    // C++:  void cv::Sobel(Mat src, Mat& dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
    //

    /**
     * Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
     *
     * In all cases except one, the \(\texttt{ksize} \times \texttt{ksize}\) separable kernel is used to
     * calculate the derivative. When \(\texttt{ksize = 1}\), the \(3 \times 1\) or \(1 \times 3\)
     * kernel is used (that is, no Gaussian smoothing is done). {@code ksize = 1} can only be used for the first
     * or the second x- or y- derivatives.
     *
     * There is also the special value {@code ksize = #FILTER_SCHARR (-1)} that corresponds to the \(3\times3\) Scharr
     * filter that may give more accurate results than the \(3\times3\) Sobel. The Scharr aperture is
     *
     * \(\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\)
     *
     * for the x-derivative, or transposed for the y-derivative.
     *
     * The function calculates an image derivative by convolving the image with the appropriate kernel:
     *
     * \(\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\)
     *
     * The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
     * resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
     * or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
     * case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\)
     *
     * The second case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src .
     * @param ddepth output image depth, see REF: filter_depths "combinations"; in the case of
     *     8-bit input images it will result in truncated derivatives.
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * @param ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
     * @param scale optional scale factor for the computed derivative values; by default, no scaling is
     * applied (see #getDerivKernels for details).
     * @param delta optional delta value that is added to the results prior to storing them in dst.
     * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE:  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar
     */
    public static void Sobel(Mat src, Mat dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType) {
        Sobel_0(src.nativeObj, dst.nativeObj, ddepth, dx, dy, ksize, scale, delta, borderType);
    }

    /**
     * Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
     *
     * In all cases except one, the \(\texttt{ksize} \times \texttt{ksize}\) separable kernel is used to
     * calculate the derivative. When \(\texttt{ksize = 1}\), the \(3 \times 1\) or \(1 \times 3\)
     * kernel is used (that is, no Gaussian smoothing is done). {@code ksize = 1} can only be used for the first
     * or the second x- or y- derivatives.
     *
     * There is also the special value {@code ksize = #FILTER_SCHARR (-1)} that corresponds to the \(3\times3\) Scharr
     * filter that may give more accurate results than the \(3\times3\) Sobel. The Scharr aperture is
     *
     * \(\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\)
     *
     * for the x-derivative, or transposed for the y-derivative.
     *
     * The function calculates an image derivative by convolving the image with the appropriate kernel:
     *
     * \(\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\)
     *
     * The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
     * resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
     * or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
     * case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\)
     *
     * The second case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src .
     * @param ddepth output image depth, see REF: filter_depths "combinations"; in the case of
     *     8-bit input images it will result in truncated derivatives.
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * @param ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
     * @param scale optional scale factor for the computed derivative values; by default, no scaling is
     * applied (see #getDerivKernels for details).
     * @param delta optional delta value that is added to the results prior to storing them in dst.
     * SEE:  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar
     */
    public static void Sobel(Mat src, Mat dst, int ddepth, int dx, int dy, int ksize, double scale, double delta) {
        Sobel_1(src.nativeObj, dst.nativeObj, ddepth, dx, dy, ksize, scale, delta);
    }

    /**
     * Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
     *
     * In all cases except one, the \(\texttt{ksize} \times \texttt{ksize}\) separable kernel is used to
     * calculate the derivative. When \(\texttt{ksize = 1}\), the \(3 \times 1\) or \(1 \times 3\)
     * kernel is used (that is, no Gaussian smoothing is done). {@code ksize = 1} can only be used for the first
     * or the second x- or y- derivatives.
     *
     * There is also the special value {@code ksize = #FILTER_SCHARR (-1)} that corresponds to the \(3\times3\) Scharr
     * filter that may give more accurate results than the \(3\times3\) Sobel. The Scharr aperture is
     *
     * \(\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\)
     *
     * for the x-derivative, or transposed for the y-derivative.
     *
     * The function calculates an image derivative by convolving the image with the appropriate kernel:
     *
     * \(\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\)
     *
     * The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
     * resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
     * or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
     * case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\)
     *
     * The second case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src .
     * @param ddepth output image depth, see REF: filter_depths "combinations"; in the case of
     *     8-bit input images it will result in truncated derivatives.
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * @param ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
     * @param scale optional scale factor for the computed derivative values; by default, no scaling is
     * applied (see #getDerivKernels for details).
     * SEE:  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar
     */
    public static void Sobel(Mat src, Mat dst, int ddepth, int dx, int dy, int ksize, double scale) {
        Sobel_2(src.nativeObj, dst.nativeObj, ddepth, dx, dy, ksize, scale);
    }

    /**
     * Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
     *
     * In all cases except one, the \(\texttt{ksize} \times \texttt{ksize}\) separable kernel is used to
     * calculate the derivative. When \(\texttt{ksize = 1}\), the \(3 \times 1\) or \(1 \times 3\)
     * kernel is used (that is, no Gaussian smoothing is done). {@code ksize = 1} can only be used for the first
     * or the second x- or y- derivatives.
     *
     * There is also the special value {@code ksize = #FILTER_SCHARR (-1)} that corresponds to the \(3\times3\) Scharr
     * filter that may give more accurate results than the \(3\times3\) Sobel. The Scharr aperture is
     *
     * \(\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\)
     *
     * for the x-derivative, or transposed for the y-derivative.
     *
     * The function calculates an image derivative by convolving the image with the appropriate kernel:
     *
     * \(\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\)
     *
     * The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
     * resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
     * or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
     * case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\)
     *
     * The second case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src .
     * @param ddepth output image depth, see REF: filter_depths "combinations"; in the case of
     *     8-bit input images it will result in truncated derivatives.
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * @param ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
     * applied (see #getDerivKernels for details).
     * SEE:  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar
     */
    public static void Sobel(Mat src, Mat dst, int ddepth, int dx, int dy, int ksize) {
        Sobel_3(src.nativeObj, dst.nativeObj, ddepth, dx, dy, ksize);
    }

    /**
     * Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
     *
     * In all cases except one, the \(\texttt{ksize} \times \texttt{ksize}\) separable kernel is used to
     * calculate the derivative. When \(\texttt{ksize = 1}\), the \(3 \times 1\) or \(1 \times 3\)
     * kernel is used (that is, no Gaussian smoothing is done). {@code ksize = 1} can only be used for the first
     * or the second x- or y- derivatives.
     *
     * There is also the special value {@code ksize = #FILTER_SCHARR (-1)} that corresponds to the \(3\times3\) Scharr
     * filter that may give more accurate results than the \(3\times3\) Sobel. The Scharr aperture is
     *
     * \(\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\)
     *
     * for the x-derivative, or transposed for the y-derivative.
     *
     * The function calculates an image derivative by convolving the image with the appropriate kernel:
     *
     * \(\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\)
     *
     * The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
     * resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
     * or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
     * case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\)
     *
     * The second case corresponds to a kernel of:
     *
     * \(\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\)
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src .
     * @param ddepth output image depth, see REF: filter_depths "combinations"; in the case of
     *     8-bit input images it will result in truncated derivatives.
     * @param dx order of the derivative x.
     * @param dy order of the derivative y.
     * applied (see #getDerivKernels for details).
     * SEE:  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar
     */
    public static void Sobel(Mat src, Mat dst, int ddepth, int dx, int dy) {
        Sobel_4(src.nativeObj, dst.nativeObj, ddepth, dx, dy);
    }


    //
    // C++:  void cv::accumulate(Mat src, Mat& dst, Mat mask = Mat())
    //

    /**
     * Adds an image to the accumulator image.
     *
     * The function adds src or some of its elements to dst :
     *
     * \(\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\)
     *
     * The function supports multi-channel images. Each channel is processed independently.
     *
     * The function cv::accumulate can be used, for example, to collect statistics of a scene background
     * viewed by a still camera and for the further foreground-background segmentation.
     *
     * @param src Input image of type CV_8UC(n), CV_16UC(n), CV_32FC(n) or CV_64FC(n), where n is a positive integer.
     * @param dst %Accumulator image with the same number of channels as input image, and a depth of CV_32F or CV_64F.
     * @param mask Optional operation mask.
     *
     * SEE:  accumulateSquare, accumulateProduct, accumulateWeighted
     */
    public static void accumulate(Mat src, Mat dst, Mat mask) {
        accumulate_0(src.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     * Adds an image to the accumulator image.
     *
     * The function adds src or some of its elements to dst :
     *
     * \(\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\)
     *
     * The function supports multi-channel images. Each channel is processed independently.
     *
     * The function cv::accumulate can be used, for example, to collect statistics of a scene background
     * viewed by a still camera and for the further foreground-background segmentation.
     *
     * @param src Input image of type CV_8UC(n), CV_16UC(n), CV_32FC(n) or CV_64FC(n), where n is a positive integer.
     * @param dst %Accumulator image with the same number of channels as input image, and a depth of CV_32F or CV_64F.
     *
     * SEE:  accumulateSquare, accumulateProduct, accumulateWeighted
     */
    public static void accumulate(Mat src, Mat dst) {
        accumulate_1(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::accumulateProduct(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
    //

    /**
     * Adds the per-element product of two input images to the accumulator image.
     *
     * The function adds the product of two images or their selected regions to the accumulator dst :
     *
     * \(\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\)
     *
     * The function supports multi-channel images. Each channel is processed independently.
     *
     * @param src1 First input image, 1- or 3-channel, 8-bit or 32-bit floating point.
     * @param src2 Second input image of the same type and the same size as src1 .
     * @param dst %Accumulator image with the same number of channels as input images, 32-bit or 64-bit
     * floating-point.
     * @param mask Optional operation mask.
     *
     * SEE:  accumulate, accumulateSquare, accumulateWeighted
     */
    public static void accumulateProduct(Mat src1, Mat src2, Mat dst, Mat mask) {
        accumulateProduct_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     * Adds the per-element product of two input images to the accumulator image.
     *
     * The function adds the product of two images or their selected regions to the accumulator dst :
     *
     * \(\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\)
     *
     * The function supports multi-channel images. Each channel is processed independently.
     *
     * @param src1 First input image, 1- or 3-channel, 8-bit or 32-bit floating point.
     * @param src2 Second input image of the same type and the same size as src1 .
     * @param dst %Accumulator image with the same number of channels as input images, 32-bit or 64-bit
     * floating-point.
     *
     * SEE:  accumulate, accumulateSquare, accumulateWeighted
     */
    public static void accumulateProduct(Mat src1, Mat src2, Mat dst) {
        accumulateProduct_1(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::accumulateSquare(Mat src, Mat& dst, Mat mask = Mat())
    //

    /**
     * Adds the square of a source image to the accumulator image.
     *
     * The function adds the input image src or its selected region, raised to a power of 2, to the
     * accumulator dst :
     *
     * \(\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\)
     *
     * The function supports multi-channel images. Each channel is processed independently.
     *
     * @param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
     * @param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit
     * floating-point.
     * @param mask Optional operation mask.
     *
     * SEE:  accumulateSquare, accumulateProduct, accumulateWeighted
     */
    public static void accumulateSquare(Mat src, Mat dst, Mat mask) {
        accumulateSquare_0(src.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     * Adds the square of a source image to the accumulator image.
     *
     * The function adds the input image src or its selected region, raised to a power of 2, to the
     * accumulator dst :
     *
     * \(\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\)
     *
     * The function supports multi-channel images. Each channel is processed independently.
     *
     * @param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
     * @param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit
     * floating-point.
     *
     * SEE:  accumulateSquare, accumulateProduct, accumulateWeighted
     */
    public static void accumulateSquare(Mat src, Mat dst) {
        accumulateSquare_1(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::accumulateWeighted(Mat src, Mat& dst, double alpha, Mat mask = Mat())
    //

    /**
     * Updates a running average.
     *
     * The function calculates the weighted sum of the input image src and the accumulator dst so that dst
     * becomes a running average of a frame sequence:
     *
     * \(\texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\)
     *
     * That is, alpha regulates the update speed (how fast the accumulator "forgets" about earlier images).
     * The function supports multi-channel images. Each channel is processed independently.
     *
     * @param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
     * @param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit
     * floating-point.
     * @param alpha Weight of the input image.
     * @param mask Optional operation mask.
     *
     * SEE:  accumulate, accumulateSquare, accumulateProduct
     */
    public static void accumulateWeighted(Mat src, Mat dst, double alpha, Mat mask) {
        accumulateWeighted_0(src.nativeObj, dst.nativeObj, alpha, mask.nativeObj);
    }

    /**
     * Updates a running average.
     *
     * The function calculates the weighted sum of the input image src and the accumulator dst so that dst
     * becomes a running average of a frame sequence:
     *
     * \(\texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\)
     *
     * That is, alpha regulates the update speed (how fast the accumulator "forgets" about earlier images).
     * The function supports multi-channel images. Each channel is processed independently.
     *
     * @param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
     * @param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit
     * floating-point.
     * @param alpha Weight of the input image.
     *
     * SEE:  accumulate, accumulateSquare, accumulateProduct
     */
    public static void accumulateWeighted(Mat src, Mat dst, double alpha) {
        accumulateWeighted_1(src.nativeObj, dst.nativeObj, alpha);
    }


    //
    // C++:  void cv::adaptiveThreshold(Mat src, Mat& dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C)
    //

    /**
     * Applies an adaptive threshold to an array.
     *
     * The function transforms a grayscale image to a binary image according to the formulae:
     * <ul>
     *   <li>
     *    <b>THRESH_BINARY</b>
     *     \(dst(x,y) =  \fork{\texttt{maxValue}}{if \(src(x,y) &gt; T(x,y)\)}{0}{otherwise}\)
     *   </li>
     *   <li>
     *    <b>THRESH_BINARY_INV</b>
     *     \(dst(x,y) =  \fork{0}{if \(src(x,y) &gt; T(x,y)\)}{\texttt{maxValue}}{otherwise}\)
     * where \(T(x,y)\) is a threshold calculated individually for each pixel (see adaptiveMethod parameter).
     *   </li>
     * </ul>
     *
     * The function can process the image in-place.
     *
     * @param src Source 8-bit single-channel image.
     * @param dst Destination image of the same size and the same type as src.
     * @param maxValue Non-zero value assigned to the pixels for which the condition is satisfied
     * @param adaptiveMethod Adaptive thresholding algorithm to use, see #AdaptiveThresholdTypes.
     * The #BORDER_REPLICATE | #BORDER_ISOLATED is used to process boundaries.
     * @param thresholdType Thresholding type that must be either #THRESH_BINARY or #THRESH_BINARY_INV,
     * see #ThresholdTypes.
     * @param blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the
     * pixel: 3, 5, 7, and so on.
     * @param C Constant subtracted from the mean or weighted mean (see the details below). Normally, it
     * is positive but may be zero or negative as well.
     *
     * SEE:  threshold, blur, GaussianBlur
     */
    public static void adaptiveThreshold(Mat src, Mat dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C) {
        adaptiveThreshold_0(src.nativeObj, dst.nativeObj, maxValue, adaptiveMethod, thresholdType, blockSize, C);
    }


    //
    // C++:  void cv::applyColorMap(Mat src, Mat& dst, Mat userColor)
    //

    /**
     * Applies a user colormap on a given image.
     *
     * @param src The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.
     * @param dst The result is the colormapped source image. Note: Mat::create is called on dst.
     * @param userColor The colormap to apply of type CV_8UC1 or CV_8UC3 and size 256
     */
    public static void applyColorMap(Mat src, Mat dst, Mat userColor) {
        applyColorMap_0(src.nativeObj, dst.nativeObj, userColor.nativeObj);
    }


    //
    // C++:  void cv::applyColorMap(Mat src, Mat& dst, int colormap)
    //

    /**
     * Applies a GNU Octave/MATLAB equivalent colormap on a given image.
     *
     * @param src The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.
     * @param dst The result is the colormapped source image. Note: Mat::create is called on dst.
     * @param colormap The colormap to apply, see #ColormapTypes
     */
    public static void applyColorMap(Mat src, Mat dst, int colormap) {
        applyColorMap_1(src.nativeObj, dst.nativeObj, colormap);
    }


    //
    // C++:  void cv::approxPolyDP(vector_Point2f curve, vector_Point2f& approxCurve, double epsilon, bool closed)
    //

    /**
     * Approximates a polygonal curve(s) with the specified precision.
     *
     * The function cv::approxPolyDP approximates a curve or a polygon with another curve/polygon with less
     * vertices so that the distance between them is less or equal to the specified precision. It uses the
     * Douglas-Peucker algorithm &lt;http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm&gt;
     *
     * @param curve Input vector of a 2D point stored in std::vector or Mat
     * @param approxCurve Result of the approximation. The type should match the type of the input curve.
     * @param epsilon Parameter specifying the approximation accuracy. This is the maximum distance
     * between the original curve and its approximation.
     * @param closed If true, the approximated curve is closed (its first and last vertices are
     * connected). Otherwise, it is not closed.
     */
    public static void approxPolyDP(MatOfPoint2f curve, MatOfPoint2f approxCurve, double epsilon, boolean closed) {
        Mat curve_mat = curve;
        Mat approxCurve_mat = approxCurve;
        approxPolyDP_0(curve_mat.nativeObj, approxCurve_mat.nativeObj, epsilon, closed);
    }


    //
    // C++:  void cv::arrowedLine(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1)
    //

    /**
     * Draws a arrow segment pointing from the first point to the second one.
     *
     * The function cv::arrowedLine draws an arrow between pt1 and pt2 points in the image. See also #line.
     *
     * @param img Image.
     * @param pt1 The point the arrow starts from.
     * @param pt2 The point the arrow points to.
     * @param color Line color.
     * @param thickness Line thickness.
     * @param line_type Type of the line. See #LineTypes
     * @param shift Number of fractional bits in the point coordinates.
     * @param tipLength The length of the arrow tip in relation to the arrow length
     */
    public static void arrowedLine(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int line_type, int shift, double tipLength) {
        arrowedLine_0(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness, line_type, shift, tipLength);
    }

    /**
     * Draws a arrow segment pointing from the first point to the second one.
     *
     * The function cv::arrowedLine draws an arrow between pt1 and pt2 points in the image. See also #line.
     *
     * @param img Image.
     * @param pt1 The point the arrow starts from.
     * @param pt2 The point the arrow points to.
     * @param color Line color.
     * @param thickness Line thickness.
     * @param line_type Type of the line. See #LineTypes
     * @param shift Number of fractional bits in the point coordinates.
     */
    public static void arrowedLine(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int line_type, int shift) {
        arrowedLine_1(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness, line_type, shift);
    }

    /**
     * Draws a arrow segment pointing from the first point to the second one.
     *
     * The function cv::arrowedLine draws an arrow between pt1 and pt2 points in the image. See also #line.
     *
     * @param img Image.
     * @param pt1 The point the arrow starts from.
     * @param pt2 The point the arrow points to.
     * @param color Line color.
     * @param thickness Line thickness.
     * @param line_type Type of the line. See #LineTypes
     */
    public static void arrowedLine(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int line_type) {
        arrowedLine_2(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness, line_type);
    }

    /**
     * Draws a arrow segment pointing from the first point to the second one.
     *
     * The function cv::arrowedLine draws an arrow between pt1 and pt2 points in the image. See also #line.
     *
     * @param img Image.
     * @param pt1 The point the arrow starts from.
     * @param pt2 The point the arrow points to.
     * @param color Line color.
     * @param thickness Line thickness.
     */
    public static void arrowedLine(Mat img, Point pt1, Point pt2, Scalar color, int thickness) {
        arrowedLine_3(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     * Draws a arrow segment pointing from the first point to the second one.
     *
     * The function cv::arrowedLine draws an arrow between pt1 and pt2 points in the image. See also #line.
     *
     * @param img Image.
     * @param pt1 The point the arrow starts from.
     * @param pt2 The point the arrow points to.
     * @param color Line color.
     */
    public static void arrowedLine(Mat img, Point pt1, Point pt2, Scalar color) {
        arrowedLine_4(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::bilateralFilter(Mat src, Mat& dst, int d, double sigmaColor, double sigmaSpace, int borderType = BORDER_DEFAULT)
    //

    /**
     * Applies the bilateral filter to an image.
     *
     * The function applies bilateral filtering to the input image, as described in
     * http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
     * bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is
     * very slow compared to most filters.
     *
     * _Sigma values_: For simplicity, you can set the 2 sigma values to be the same. If they are small (&lt;
     * 10), the filter will not have much effect, whereas if they are large (&gt; 150), they will have a very
     * strong effect, making the image look "cartoonish".
     *
     * _Filter size_: Large filters (d &gt; 5) are very slow, so it is recommended to use d=5 for real-time
     * applications, and perhaps d=9 for offline applications that need heavy noise filtering.
     *
     * This filter does not work inplace.
     * @param src Source 8-bit or floating-point, 1-channel or 3-channel image.
     * @param dst Destination image of the same size and type as src .
     * @param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
     * it is computed from sigmaSpace.
     * @param sigmaColor Filter sigma in the color space. A larger value of the parameter means that
     * farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting
     * in larger areas of semi-equal color.
     * @param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that
     * farther pixels will influence each other as long as their colors are close enough (see sigmaColor
     * ). When d&gt;0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is
     * proportional to sigmaSpace.
     * @param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes
     */
    public static void bilateralFilter(Mat src, Mat dst, int d, double sigmaColor, double sigmaSpace, int borderType) {
        bilateralFilter_0(src.nativeObj, dst.nativeObj, d, sigmaColor, sigmaSpace, borderType);
    }

    /**
     * Applies the bilateral filter to an image.
     *
     * The function applies bilateral filtering to the input image, as described in
     * http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
     * bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is
     * very slow compared to most filters.
     *
     * _Sigma values_: For simplicity, you can set the 2 sigma values to be the same. If they are small (&lt;
     * 10), the filter will not have much effect, whereas if they are large (&gt; 150), they will have a very
     * strong effect, making the image look "cartoonish".
     *
     * _Filter size_: Large filters (d &gt; 5) are very slow, so it is recommended to use d=5 for real-time
     * applications, and perhaps d=9 for offline applications that need heavy noise filtering.
     *
     * This filter does not work inplace.
     * @param src Source 8-bit or floating-point, 1-channel or 3-channel image.
     * @param dst Destination image of the same size and type as src .
     * @param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
     * it is computed from sigmaSpace.
     * @param sigmaColor Filter sigma in the color space. A larger value of the parameter means that
     * farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting
     * in larger areas of semi-equal color.
     * @param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that
     * farther pixels will influence each other as long as their colors are close enough (see sigmaColor
     * ). When d&gt;0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is
     * proportional to sigmaSpace.
     */
    public static void bilateralFilter(Mat src, Mat dst, int d, double sigmaColor, double sigmaSpace) {
        bilateralFilter_1(src.nativeObj, dst.nativeObj, d, sigmaColor, sigmaSpace);
    }


    //
    // C++:  void cv::blur(Mat src, Mat& dst, Size ksize, Point anchor = Point(-1,-1), int borderType = BORDER_DEFAULT)
    //

    /**
     * Blurs an image using the normalized box filter.
     *
     * The function smooths an image using the kernel:
     *
     * \(\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \hdotsfor{6} \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \end{bmatrix}\)
     *
     * The call {@code blur(src, dst, ksize, anchor, borderType)} is equivalent to `boxFilter(src, dst, src.type(),
     * anchor, true, borderType)`.
     *
     * @param src input image; it can have any number of channels, which are processed independently, but
     * the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param ksize blurring kernel size.
     * @param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
     * center.
     * @param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE:  boxFilter, bilateralFilter, GaussianBlur, medianBlur
     */
    public static void blur(Mat src, Mat dst, Size ksize, Point anchor, int borderType) {
        blur_0(src.nativeObj, dst.nativeObj, ksize.width, ksize.height, anchor.x, anchor.y, borderType);
    }

    /**
     * Blurs an image using the normalized box filter.
     *
     * The function smooths an image using the kernel:
     *
     * \(\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \hdotsfor{6} \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \end{bmatrix}\)
     *
     * The call {@code blur(src, dst, ksize, anchor, borderType)} is equivalent to `boxFilter(src, dst, src.type(),
     * anchor, true, borderType)`.
     *
     * @param src input image; it can have any number of channels, which are processed independently, but
     * the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param ksize blurring kernel size.
     * @param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
     * center.
     * SEE:  boxFilter, bilateralFilter, GaussianBlur, medianBlur
     */
    public static void blur(Mat src, Mat dst, Size ksize, Point anchor) {
        blur_1(src.nativeObj, dst.nativeObj, ksize.width, ksize.height, anchor.x, anchor.y);
    }

    /**
     * Blurs an image using the normalized box filter.
     *
     * The function smooths an image using the kernel:
     *
     * \(\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \hdotsfor{6} \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \end{bmatrix}\)
     *
     * The call {@code blur(src, dst, ksize, anchor, borderType)} is equivalent to `boxFilter(src, dst, src.type(),
     * anchor, true, borderType)`.
     *
     * @param src input image; it can have any number of channels, which are processed independently, but
     * the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param ksize blurring kernel size.
     * center.
     * SEE:  boxFilter, bilateralFilter, GaussianBlur, medianBlur
     */
    public static void blur(Mat src, Mat dst, Size ksize) {
        blur_2(src.nativeObj, dst.nativeObj, ksize.width, ksize.height);
    }


    //
    // C++:  void cv::boxFilter(Mat src, Mat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), bool normalize = true, int borderType = BORDER_DEFAULT)
    //

    /**
     * Blurs an image using the box filter.
     *
     * The function smooths an image using the kernel:
     *
     * \(\texttt{K} =  \alpha \begin{bmatrix} 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \hdotsfor{6} \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1 \end{bmatrix}\)
     *
     * where
     *
     * \(\alpha = \fork{\frac{1}{\texttt{ksize.width*ksize.height}}}{when \texttt{normalize=true}}{1}{otherwise}\)
     *
     * Unnormalized box filter is useful for computing various integral characteristics over each pixel
     * neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
     * algorithms, and so on). If you need to compute pixel sums over variable-size windows, use #integral.
     *
     * @param src input image.
     * @param dst output image of the same size and type as src.
     * @param ddepth the output image depth (-1 to use src.depth()).
     * @param ksize blurring kernel size.
     * @param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
     * center.
     * @param normalize flag, specifying whether the kernel is normalized by its area or not.
     * @param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE:  blur, bilateralFilter, GaussianBlur, medianBlur, integral
     */
    public static void boxFilter(Mat src, Mat dst, int ddepth, Size ksize, Point anchor, boolean normalize, int borderType) {
        boxFilter_0(src.nativeObj, dst.nativeObj, ddepth, ksize.width, ksize.height, anchor.x, anchor.y, normalize, borderType);
    }

    /**
     * Blurs an image using the box filter.
     *
     * The function smooths an image using the kernel:
     *
     * \(\texttt{K} =  \alpha \begin{bmatrix} 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \hdotsfor{6} \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1 \end{bmatrix}\)
     *
     * where
     *
     * \(\alpha = \fork{\frac{1}{\texttt{ksize.width*ksize.height}}}{when \texttt{normalize=true}}{1}{otherwise}\)
     *
     * Unnormalized box filter is useful for computing various integral characteristics over each pixel
     * neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
     * algorithms, and so on). If you need to compute pixel sums over variable-size windows, use #integral.
     *
     * @param src input image.
     * @param dst output image of the same size and type as src.
     * @param ddepth the output image depth (-1 to use src.depth()).
     * @param ksize blurring kernel size.
     * @param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
     * center.
     * @param normalize flag, specifying whether the kernel is normalized by its area or not.
     * SEE:  blur, bilateralFilter, GaussianBlur, medianBlur, integral
     */
    public static void boxFilter(Mat src, Mat dst, int ddepth, Size ksize, Point anchor, boolean normalize) {
        boxFilter_1(src.nativeObj, dst.nativeObj, ddepth, ksize.width, ksize.height, anchor.x, anchor.y, normalize);
    }

    /**
     * Blurs an image using the box filter.
     *
     * The function smooths an image using the kernel:
     *
     * \(\texttt{K} =  \alpha \begin{bmatrix} 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \hdotsfor{6} \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1 \end{bmatrix}\)
     *
     * where
     *
     * \(\alpha = \fork{\frac{1}{\texttt{ksize.width*ksize.height}}}{when \texttt{normalize=true}}{1}{otherwise}\)
     *
     * Unnormalized box filter is useful for computing various integral characteristics over each pixel
     * neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
     * algorithms, and so on). If you need to compute pixel sums over variable-size windows, use #integral.
     *
     * @param src input image.
     * @param dst output image of the same size and type as src.
     * @param ddepth the output image depth (-1 to use src.depth()).
     * @param ksize blurring kernel size.
     * @param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
     * center.
     * SEE:  blur, bilateralFilter, GaussianBlur, medianBlur, integral
     */
    public static void boxFilter(Mat src, Mat dst, int ddepth, Size ksize, Point anchor) {
        boxFilter_2(src.nativeObj, dst.nativeObj, ddepth, ksize.width, ksize.height, anchor.x, anchor.y);
    }

    /**
     * Blurs an image using the box filter.
     *
     * The function smooths an image using the kernel:
     *
     * \(\texttt{K} =  \alpha \begin{bmatrix} 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1  \\ \hdotsfor{6} \\ 1 &amp; 1 &amp; 1 &amp;  \cdots &amp; 1 &amp; 1 \end{bmatrix}\)
     *
     * where
     *
     * \(\alpha = \fork{\frac{1}{\texttt{ksize.width*ksize.height}}}{when \texttt{normalize=true}}{1}{otherwise}\)
     *
     * Unnormalized box filter is useful for computing various integral characteristics over each pixel
     * neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
     * algorithms, and so on). If you need to compute pixel sums over variable-size windows, use #integral.
     *
     * @param src input image.
     * @param dst output image of the same size and type as src.
     * @param ddepth the output image depth (-1 to use src.depth()).
     * @param ksize blurring kernel size.
     * center.
     * SEE:  blur, bilateralFilter, GaussianBlur, medianBlur, integral
     */
    public static void boxFilter(Mat src, Mat dst, int ddepth, Size ksize) {
        boxFilter_3(src.nativeObj, dst.nativeObj, ddepth, ksize.width, ksize.height);
    }


    //
    // C++:  void cv::boxPoints(RotatedRect box, Mat& points)
    //

    /**
     * Finds the four vertices of a rotated rect. Useful to draw the rotated rectangle.
     *
     * The function finds the four vertices of a rotated rectangle. This function is useful to draw the
     * rectangle. In C++, instead of using this function, you can directly use RotatedRect::points method. Please
     * visit the REF: tutorial_bounding_rotated_ellipses "tutorial on Creating Bounding rotated boxes and ellipses for contours" for more information.
     *
     * @param box The input rotated rectangle. It may be the output of
     * @param points The output array of four vertices of rectangles.
     */
    public static void boxPoints(RotatedRect box, Mat points) {
        boxPoints_0(box.center.x, box.center.y, box.size.width, box.size.height, box.angle, points.nativeObj);
    }


    //
    // C++:  void cv::calcBackProject(vector_Mat images, vector_int channels, Mat hist, Mat& dst, vector_float ranges, double scale)
    //

    public static void calcBackProject(List<Mat> images, MatOfInt channels, Mat hist, Mat dst, MatOfFloat ranges, double scale) {
        Mat images_mat = Converters.vector_Mat_to_Mat(images);
        Mat channels_mat = channels;
        Mat ranges_mat = ranges;
        calcBackProject_0(images_mat.nativeObj, channels_mat.nativeObj, hist.nativeObj, dst.nativeObj, ranges_mat.nativeObj, scale);
    }


    //
    // C++:  void cv::calcHist(vector_Mat images, vector_int channels, Mat mask, Mat& hist, vector_int histSize, vector_float ranges, bool accumulate = false)
    //

    public static void calcHist(List<Mat> images, MatOfInt channels, Mat mask, Mat hist, MatOfInt histSize, MatOfFloat ranges, boolean accumulate) {
        Mat images_mat = Converters.vector_Mat_to_Mat(images);
        Mat channels_mat = channels;
        Mat histSize_mat = histSize;
        Mat ranges_mat = ranges;
        calcHist_0(images_mat.nativeObj, channels_mat.nativeObj, mask.nativeObj, hist.nativeObj, histSize_mat.nativeObj, ranges_mat.nativeObj, accumulate);
    }

    public static void calcHist(List<Mat> images, MatOfInt channels, Mat mask, Mat hist, MatOfInt histSize, MatOfFloat ranges) {
        Mat images_mat = Converters.vector_Mat_to_Mat(images);
        Mat channels_mat = channels;
        Mat histSize_mat = histSize;
        Mat ranges_mat = ranges;
        calcHist_1(images_mat.nativeObj, channels_mat.nativeObj, mask.nativeObj, hist.nativeObj, histSize_mat.nativeObj, ranges_mat.nativeObj);
    }


    //
    // C++:  void cv::circle(Mat& img, Point center, int radius, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    //

    /**
     * Draws a circle.
     *
     * The function cv::circle draws a simple or filled circle with a given center and radius.
     * @param img Image where the circle is drawn.
     * @param center Center of the circle.
     * @param radius Radius of the circle.
     * @param color Circle color.
     * @param thickness Thickness of the circle outline, if positive. Negative values, like #FILLED,
     * mean that a filled circle is to be drawn.
     * @param lineType Type of the circle boundary. See #LineTypes
     * @param shift Number of fractional bits in the coordinates of the center and in the radius value.
     */
    public static void circle(Mat img, Point center, int radius, Scalar color, int thickness, int lineType, int shift) {
        circle_0(img.nativeObj, center.x, center.y, radius, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, shift);
    }

    /**
     * Draws a circle.
     *
     * The function cv::circle draws a simple or filled circle with a given center and radius.
     * @param img Image where the circle is drawn.
     * @param center Center of the circle.
     * @param radius Radius of the circle.
     * @param color Circle color.
     * @param thickness Thickness of the circle outline, if positive. Negative values, like #FILLED,
     * mean that a filled circle is to be drawn.
     * @param lineType Type of the circle boundary. See #LineTypes
     */
    public static void circle(Mat img, Point center, int radius, Scalar color, int thickness, int lineType) {
        circle_1(img.nativeObj, center.x, center.y, radius, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     * Draws a circle.
     *
     * The function cv::circle draws a simple or filled circle with a given center and radius.
     * @param img Image where the circle is drawn.
     * @param center Center of the circle.
     * @param radius Radius of the circle.
     * @param color Circle color.
     * @param thickness Thickness of the circle outline, if positive. Negative values, like #FILLED,
     * mean that a filled circle is to be drawn.
     */
    public static void circle(Mat img, Point center, int radius, Scalar color, int thickness) {
        circle_2(img.nativeObj, center.x, center.y, radius, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     * Draws a circle.
     *
     * The function cv::circle draws a simple or filled circle with a given center and radius.
     * @param img Image where the circle is drawn.
     * @param center Center of the circle.
     * @param radius Radius of the circle.
     * @param color Circle color.
     * mean that a filled circle is to be drawn.
     */
    public static void circle(Mat img, Point center, int radius, Scalar color) {
        circle_3(img.nativeObj, center.x, center.y, radius, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::convertMaps(Mat map1, Mat map2, Mat& dstmap1, Mat& dstmap2, int dstmap1type, bool nninterpolation = false)
    //

    /**
     * Converts image transformation maps from one representation to another.
     *
     * The function converts a pair of maps for remap from one representation to another. The following
     * options ( (map1.type(), map2.type()) \(\rightarrow\) (dstmap1.type(), dstmap2.type()) ) are
     * supported:
     *
     * <ul>
     *   <li>
     *  \(\texttt{(CV_32FC1, CV_32FC1)} \rightarrow \texttt{(CV_16SC2, CV_16UC1)}\). This is the
     * most frequently used conversion operation, in which the original floating-point maps (see remap )
     * are converted to a more compact and much faster fixed-point representation. The first output array
     * contains the rounded coordinates and the second array (created only when nninterpolation=false )
     * contains indices in the interpolation tables.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *  \(\texttt{(CV_32FC2)} \rightarrow \texttt{(CV_16SC2, CV_16UC1)}\). The same as above but
     * the original maps are stored in one 2-channel matrix.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *  Reverse conversion. Obviously, the reconstructed floating-point maps will not be exactly the same
     * as the originals.
     *   </li>
     * </ul>
     *
     * @param map1 The first input map of type CV_16SC2, CV_32FC1, or CV_32FC2 .
     * @param map2 The second input map of type CV_16UC1, CV_32FC1, or none (empty matrix),
     * respectively.
     * @param dstmap1 The first output map that has the type dstmap1type and the same size as src .
     * @param dstmap2 The second output map.
     * @param dstmap1type Type of the first output map that should be CV_16SC2, CV_32FC1, or
     * CV_32FC2 .
     * @param nninterpolation Flag indicating whether the fixed-point maps are used for the
     * nearest-neighbor or for a more complex interpolation.
     *
     * SEE:  remap, undistort, initUndistortRectifyMap
     */
    public static void convertMaps(Mat map1, Mat map2, Mat dstmap1, Mat dstmap2, int dstmap1type, boolean nninterpolation) {
        convertMaps_0(map1.nativeObj, map2.nativeObj, dstmap1.nativeObj, dstmap2.nativeObj, dstmap1type, nninterpolation);
    }

    /**
     * Converts image transformation maps from one representation to another.
     *
     * The function converts a pair of maps for remap from one representation to another. The following
     * options ( (map1.type(), map2.type()) \(\rightarrow\) (dstmap1.type(), dstmap2.type()) ) are
     * supported:
     *
     * <ul>
     *   <li>
     *  \(\texttt{(CV_32FC1, CV_32FC1)} \rightarrow \texttt{(CV_16SC2, CV_16UC1)}\). This is the
     * most frequently used conversion operation, in which the original floating-point maps (see remap )
     * are converted to a more compact and much faster fixed-point representation. The first output array
     * contains the rounded coordinates and the second array (created only when nninterpolation=false )
     * contains indices in the interpolation tables.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *  \(\texttt{(CV_32FC2)} \rightarrow \texttt{(CV_16SC2, CV_16UC1)}\). The same as above but
     * the original maps are stored in one 2-channel matrix.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *  Reverse conversion. Obviously, the reconstructed floating-point maps will not be exactly the same
     * as the originals.
     *   </li>
     * </ul>
     *
     * @param map1 The first input map of type CV_16SC2, CV_32FC1, or CV_32FC2 .
     * @param map2 The second input map of type CV_16UC1, CV_32FC1, or none (empty matrix),
     * respectively.
     * @param dstmap1 The first output map that has the type dstmap1type and the same size as src .
     * @param dstmap2 The second output map.
     * @param dstmap1type Type of the first output map that should be CV_16SC2, CV_32FC1, or
     * CV_32FC2 .
     * nearest-neighbor or for a more complex interpolation.
     *
     * SEE:  remap, undistort, initUndistortRectifyMap
     */
    public static void convertMaps(Mat map1, Mat map2, Mat dstmap1, Mat dstmap2, int dstmap1type) {
        convertMaps_1(map1.nativeObj, map2.nativeObj, dstmap1.nativeObj, dstmap2.nativeObj, dstmap1type);
    }


    //
    // C++:  void cv::convexHull(vector_Point points, vector_int& hull, bool clockwise = false,  _hidden_  returnPoints = true)
    //

    /**
     * Finds the convex hull of a point set.
     *
     * The function cv::convexHull finds the convex hull of a 2D point set using the Sklansky's algorithm CITE: Sklansky82
     * that has *O(N logN)* complexity in the current implementation.
     *
     * @param points Input 2D point set, stored in std::vector or Mat.
     * @param hull Output convex hull. It is either an integer vector of indices or vector of points. In
     * the first case, the hull elements are 0-based indices of the convex hull points in the original
     * array (since the set of convex hull points is a subset of the original point set). In the second
     * case, hull elements are the convex hull points themselves.
     * @param clockwise Orientation flag. If it is true, the output convex hull is oriented clockwise.
     * Otherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing
     * to the right, and its Y axis pointing upwards.
     * returns convex hull points. Otherwise, it returns indices of the convex hull points. When the
     * output array is std::vector, the flag is ignored, and the output depends on the type of the
     * vector: std::vector&lt;int&gt; implies returnPoints=false, std::vector&lt;Point&gt; implies
     * returnPoints=true.
     *
     * <b>Note:</b> {@code points} and {@code hull} should be different arrays, inplace processing isn't supported.
     *
     * Check REF: tutorial_hull "the corresponding tutorial" for more details.
     *
     * useful links:
     *
     * https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/
     */
    public static void convexHull(MatOfPoint points, MatOfInt hull, boolean clockwise) {
        Mat points_mat = points;
        Mat hull_mat = hull;
        convexHull_0(points_mat.nativeObj, hull_mat.nativeObj, clockwise);
    }

    /**
     * Finds the convex hull of a point set.
     *
     * The function cv::convexHull finds the convex hull of a 2D point set using the Sklansky's algorithm CITE: Sklansky82
     * that has *O(N logN)* complexity in the current implementation.
     *
     * @param points Input 2D point set, stored in std::vector or Mat.
     * @param hull Output convex hull. It is either an integer vector of indices or vector of points. In
     * the first case, the hull elements are 0-based indices of the convex hull points in the original
     * array (since the set of convex hull points is a subset of the original point set). In the second
     * case, hull elements are the convex hull points themselves.
     * Otherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing
     * to the right, and its Y axis pointing upwards.
     * returns convex hull points. Otherwise, it returns indices of the convex hull points. When the
     * output array is std::vector, the flag is ignored, and the output depends on the type of the
     * vector: std::vector&lt;int&gt; implies returnPoints=false, std::vector&lt;Point&gt; implies
     * returnPoints=true.
     *
     * <b>Note:</b> {@code points} and {@code hull} should be different arrays, inplace processing isn't supported.
     *
     * Check REF: tutorial_hull "the corresponding tutorial" for more details.
     *
     * useful links:
     *
     * https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/
     */
    public static void convexHull(MatOfPoint points, MatOfInt hull) {
        Mat points_mat = points;
        Mat hull_mat = hull;
        convexHull_2(points_mat.nativeObj, hull_mat.nativeObj);
    }


    //
    // C++:  void cv::convexityDefects(vector_Point contour, vector_int convexhull, vector_Vec4i& convexityDefects)
    //

    /**
     * Finds the convexity defects of a contour.
     *
     * The figure below displays convexity defects of a hand contour:
     *
     * ![image](pics/defects.png)
     *
     * @param contour Input contour.
     * @param convexhull Convex hull obtained using convexHull that should contain indices of the contour
     * points that make the hull.
     * @param convexityDefects The output vector of convexity defects. In C++ and the new Python/Java
     * interface each convexity defect is represented as 4-element integer vector (a.k.a. #Vec4i):
     * (start_index, end_index, farthest_pt_index, fixpt_depth), where indices are 0-based indices
     * in the original contour of the convexity defect beginning, end and the farthest point, and
     * fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the
     * farthest contour point and the hull. That is, to get the floating-point value of the depth will be
     * fixpt_depth/256.0.
     */
    public static void convexityDefects(MatOfPoint contour, MatOfInt convexhull, MatOfInt4 convexityDefects) {
        Mat contour_mat = contour;
        Mat convexhull_mat = convexhull;
        Mat convexityDefects_mat = convexityDefects;
        convexityDefects_0(contour_mat.nativeObj, convexhull_mat.nativeObj, convexityDefects_mat.nativeObj);
    }


    //
    // C++:  void cv::cornerEigenValsAndVecs(Mat src, Mat& dst, int blockSize, int ksize, int borderType = BORDER_DEFAULT)
    //

    /**
     * Calculates eigenvalues and eigenvectors of image blocks for corner detection.
     *
     * For every pixel \(p\) , the function cornerEigenValsAndVecs considers a blockSize \(\times\) blockSize
     * neighborhood \(S(p)\) . It calculates the covariation matrix of derivatives over the neighborhood as:
     *
     * \(M =  \begin{bmatrix} \sum _{S(p)}(dI/dx)^2 &amp;  \sum _{S(p)}dI/dx dI/dy  \\ \sum _{S(p)}dI/dx dI/dy &amp;  \sum _{S(p)}(dI/dy)^2 \end{bmatrix}\)
     *
     * where the derivatives are computed using the Sobel operator.
     *
     * After that, it finds eigenvectors and eigenvalues of \(M\) and stores them in the destination image as
     * \((\lambda_1, \lambda_2, x_1, y_1, x_2, y_2)\) where
     *
     * <ul>
     *   <li>
     *    \(\lambda_1, \lambda_2\) are the non-sorted eigenvalues of \(M\)
     *   </li>
     *   <li>
     *    \(x_1, y_1\) are the eigenvectors corresponding to \(\lambda_1\)
     *   </li>
     *   <li>
     *    \(x_2, y_2\) are the eigenvectors corresponding to \(\lambda_2\)
     *   </li>
     * </ul>
     *
     * The output of the function can be used for robust edge or corner detection.
     *
     * @param src Input single-channel 8-bit or floating-point image.
     * @param dst Image to store the results. It has the same size as src and the type CV_32FC(6) .
     * @param blockSize Neighborhood size (see details below).
     * @param ksize Aperture parameter for the Sobel operator.
     * @param borderType Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.
     *
     * SEE:  cornerMinEigenVal, cornerHarris, preCornerDetect
     */
    public static void cornerEigenValsAndVecs(Mat src, Mat dst, int blockSize, int ksize, int borderType) {
        cornerEigenValsAndVecs_0(src.nativeObj, dst.nativeObj, blockSize, ksize, borderType);
    }

    /**
     * Calculates eigenvalues and eigenvectors of image blocks for corner detection.
     *
     * For every pixel \(p\) , the function cornerEigenValsAndVecs considers a blockSize \(\times\) blockSize
     * neighborhood \(S(p)\) . It calculates the covariation matrix of derivatives over the neighborhood as:
     *
     * \(M =  \begin{bmatrix} \sum _{S(p)}(dI/dx)^2 &amp;  \sum _{S(p)}dI/dx dI/dy  \\ \sum _{S(p)}dI/dx dI/dy &amp;  \sum _{S(p)}(dI/dy)^2 \end{bmatrix}\)
     *
     * where the derivatives are computed using the Sobel operator.
     *
     * After that, it finds eigenvectors and eigenvalues of \(M\) and stores them in the destination image as
     * \((\lambda_1, \lambda_2, x_1, y_1, x_2, y_2)\) where
     *
     * <ul>
     *   <li>
     *    \(\lambda_1, \lambda_2\) are the non-sorted eigenvalues of \(M\)
     *   </li>
     *   <li>
     *    \(x_1, y_1\) are the eigenvectors corresponding to \(\lambda_1\)
     *   </li>
     *   <li>
     *    \(x_2, y_2\) are the eigenvectors corresponding to \(\lambda_2\)
     *   </li>
     * </ul>
     *
     * The output of the function can be used for robust edge or corner detection.
     *
     * @param src Input single-channel 8-bit or floating-point image.
     * @param dst Image to store the results. It has the same size as src and the type CV_32FC(6) .
     * @param blockSize Neighborhood size (see details below).
     * @param ksize Aperture parameter for the Sobel operator.
     *
     * SEE:  cornerMinEigenVal, cornerHarris, preCornerDetect
     */
    public static void cornerEigenValsAndVecs(Mat src, Mat dst, int blockSize, int ksize) {
        cornerEigenValsAndVecs_1(src.nativeObj, dst.nativeObj, blockSize, ksize);
    }


    //
    // C++:  void cv::cornerHarris(Mat src, Mat& dst, int blockSize, int ksize, double k, int borderType = BORDER_DEFAULT)
    //

    /**
     * Harris corner detector.
     *
     * The function runs the Harris corner detector on the image. Similarly to cornerMinEigenVal and
     * cornerEigenValsAndVecs , for each pixel \((x, y)\) it calculates a \(2\times2\) gradient covariance
     * matrix \(M^{(x,y)}\) over a \(\texttt{blockSize} \times \texttt{blockSize}\) neighborhood. Then, it
     * computes the following characteristic:
     *
     * \(\texttt{dst} (x,y) =  \mathrm{det} M^{(x,y)} - k  \cdot \left ( \mathrm{tr} M^{(x,y)} \right )^2\)
     *
     * Corners in the image can be found as the local maxima of this response map.
     *
     * @param src Input single-channel 8-bit or floating-point image.
     * @param dst Image to store the Harris detector responses. It has the type CV_32FC1 and the same
     * size as src .
     * @param blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).
     * @param ksize Aperture parameter for the Sobel operator.
     * @param k Harris detector free parameter. See the formula above.
     * @param borderType Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.
     */
    public static void cornerHarris(Mat src, Mat dst, int blockSize, int ksize, double k, int borderType) {
        cornerHarris_0(src.nativeObj, dst.nativeObj, blockSize, ksize, k, borderType);
    }

    /**
     * Harris corner detector.
     *
     * The function runs the Harris corner detector on the image. Similarly to cornerMinEigenVal and
     * cornerEigenValsAndVecs , for each pixel \((x, y)\) it calculates a \(2\times2\) gradient covariance
     * matrix \(M^{(x,y)}\) over a \(\texttt{blockSize} \times \texttt{blockSize}\) neighborhood. Then, it
     * computes the following characteristic:
     *
     * \(\texttt{dst} (x,y) =  \mathrm{det} M^{(x,y)} - k  \cdot \left ( \mathrm{tr} M^{(x,y)} \right )^2\)
     *
     * Corners in the image can be found as the local maxima of this response map.
     *
     * @param src Input single-channel 8-bit or floating-point image.
     * @param dst Image to store the Harris detector responses. It has the type CV_32FC1 and the same
     * size as src .
     * @param blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).
     * @param ksize Aperture parameter for the Sobel operator.
     * @param k Harris detector free parameter. See the formula above.
     */
    public static void cornerHarris(Mat src, Mat dst, int blockSize, int ksize, double k) {
        cornerHarris_1(src.nativeObj, dst.nativeObj, blockSize, ksize, k);
    }


    //
    // C++:  void cv::cornerMinEigenVal(Mat src, Mat& dst, int blockSize, int ksize = 3, int borderType = BORDER_DEFAULT)
    //

    /**
     * Calculates the minimal eigenvalue of gradient matrices for corner detection.
     *
     * The function is similar to cornerEigenValsAndVecs but it calculates and stores only the minimal
     * eigenvalue of the covariance matrix of derivatives, that is, \(\min(\lambda_1, \lambda_2)\) in terms
     * of the formulae in the cornerEigenValsAndVecs description.
     *
     * @param src Input single-channel 8-bit or floating-point image.
     * @param dst Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size as
     * src .
     * @param blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).
     * @param ksize Aperture parameter for the Sobel operator.
     * @param borderType Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.
     */
    public static void cornerMinEigenVal(Mat src, Mat dst, int blockSize, int ksize, int borderType) {
        cornerMinEigenVal_0(src.nativeObj, dst.nativeObj, blockSize, ksize, borderType);
    }

    /**
     * Calculates the minimal eigenvalue of gradient matrices for corner detection.
     *
     * The function is similar to cornerEigenValsAndVecs but it calculates and stores only the minimal
     * eigenvalue of the covariance matrix of derivatives, that is, \(\min(\lambda_1, \lambda_2)\) in terms
     * of the formulae in the cornerEigenValsAndVecs description.
     *
     * @param src Input single-channel 8-bit or floating-point image.
     * @param dst Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size as
     * src .
     * @param blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).
     * @param ksize Aperture parameter for the Sobel operator.
     */
    public static void cornerMinEigenVal(Mat src, Mat dst, int blockSize, int ksize) {
        cornerMinEigenVal_1(src.nativeObj, dst.nativeObj, blockSize, ksize);
    }

    /**
     * Calculates the minimal eigenvalue of gradient matrices for corner detection.
     *
     * The function is similar to cornerEigenValsAndVecs but it calculates and stores only the minimal
     * eigenvalue of the covariance matrix of derivatives, that is, \(\min(\lambda_1, \lambda_2)\) in terms
     * of the formulae in the cornerEigenValsAndVecs description.
     *
     * @param src Input single-channel 8-bit or floating-point image.
     * @param dst Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size as
     * src .
     * @param blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).
     */
    public static void cornerMinEigenVal(Mat src, Mat dst, int blockSize) {
        cornerMinEigenVal_2(src.nativeObj, dst.nativeObj, blockSize);
    }


    //
    // C++:  void cv::cornerSubPix(Mat image, Mat& corners, Size winSize, Size zeroZone, TermCriteria criteria)
    //

    /**
     * Refines the corner locations.
     *
     * The function iterates to find the sub-pixel accurate location of corners or radial saddle points, as
     * shown on the figure below.
     *
     * ![image](pics/cornersubpix.png)
     *
     * Sub-pixel accurate corner locator is based on the observation that every vector from the center \(q\)
     * to a point \(p\) located within a neighborhood of \(q\) is orthogonal to the image gradient at \(p\)
     * subject to image and measurement noise. Consider the expression:
     *
     * \(\epsilon _i = {DI_{p_i}}^T  \cdot (q - p_i)\)
     *
     * where \({DI_{p_i}}\) is an image gradient at one of the points \(p_i\) in a neighborhood of \(q\) . The
     * value of \(q\) is to be found so that \(\epsilon_i\) is minimized. A system of equations may be set up
     * with \(\epsilon_i\) set to zero:
     *
     * \(\sum _i(DI_{p_i}  \cdot {DI_{p_i}}^T) \cdot q -  \sum _i(DI_{p_i}  \cdot {DI_{p_i}}^T  \cdot p_i)\)
     *
     * where the gradients are summed within a neighborhood ("search window") of \(q\) . Calling the first
     * gradient term \(G\) and the second gradient term \(b\) gives:
     *
     * \(q = G^{-1}  \cdot b\)
     *
     * The algorithm sets the center of the neighborhood window at this new center \(q\) and then iterates
     * until the center stays within a set threshold.
     *
     * @param image Input single-channel, 8-bit or float image.
     * @param corners Initial coordinates of the input corners and refined coordinates provided for
     * output.
     * @param winSize Half of the side length of the search window. For example, if winSize=Size(5,5) ,
     * then a \((5*2+1) \times (5*2+1) = 11 \times 11\) search window is used.
     * @param zeroZone Half of the size of the dead region in the middle of the search zone over which
     * the summation in the formula below is not done. It is used sometimes to avoid possible
     * singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such
     * a size.
     * @param criteria Criteria for termination of the iterative process of corner refinement. That is,
     * the process of corner position refinement stops either after criteria.maxCount iterations or when
     * the corner position moves by less than criteria.epsilon on some iteration.
     */
    public static void cornerSubPix(Mat image, Mat corners, Size winSize, Size zeroZone, TermCriteria criteria) {
        cornerSubPix_0(image.nativeObj, corners.nativeObj, winSize.width, winSize.height, zeroZone.width, zeroZone.height, criteria.type, criteria.maxCount, criteria.epsilon);
    }


    //
    // C++:  void cv::createHanningWindow(Mat& dst, Size winSize, int type)
    //

    /**
     * This function computes a Hanning window coefficients in two dimensions.
     *
     * See (http://en.wikipedia.org/wiki/Hann_function) and (http://en.wikipedia.org/wiki/Window_function)
     * for more information.
     *
     * An example is shown below:
     * <code>
     *     // create hanning window of size 100x100 and type CV_32F
     *     Mat hann;
     *     createHanningWindow(hann, Size(100, 100), CV_32F);
     * </code>
     * @param dst Destination array to place Hann coefficients in
     * @param winSize The window size specifications (both width and height must be &gt; 1)
     * @param type Created array type
     */
    public static void createHanningWindow(Mat dst, Size winSize, int type) {
        createHanningWindow_0(dst.nativeObj, winSize.width, winSize.height, type);
    }


    //
    // C++:  void cv::cvtColor(Mat src, Mat& dst, int code, int dstCn = 0)
    //

    /**
     * Converts an image from one color space to another.
     *
     * The function converts an input image from one color space to another. In case of a transformation
     * to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note
     * that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the
     * bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue
     * component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and
     * sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.
     *
     * The conventional ranges for R, G, and B channel values are:
     * <ul>
     *   <li>
     *    0 to 255 for CV_8U images
     *   </li>
     *   <li>
     *    0 to 65535 for CV_16U images
     *   </li>
     *   <li>
     *    0 to 1 for CV_32F images
     *   </li>
     * </ul>
     *
     * In case of linear transformations, the range does not matter. But in case of a non-linear
     * transformation, an input RGB image should be normalized to the proper value range to get the correct
     * results, for example, for RGB \(\rightarrow\) L\*u\*v\* transformation. For example, if you have a
     * 32-bit floating-point image directly converted from an 8-bit image without any scaling, then it will
     * have the 0..255 value range instead of 0..1 assumed by the function. So, before calling #cvtColor ,
     * you need first to scale the image down:
     * <code>
     *     img *= 1./255;
     *     cvtColor(img, img, COLOR_BGR2Luv);
     * </code>
     * If you use #cvtColor with 8-bit images, the conversion will have some information lost. For many
     * applications, this will not be noticeable but it is recommended to use 32-bit images in applications
     * that need the full range of colors or that convert an image before an operation and then convert
     * back.
     *
     * If conversion adds the alpha channel, its value will set to the maximum of corresponding channel
     * range: 255 for CV_8U, 65535 for CV_16U, 1 for CV_32F.
     *
     * @param src input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision
     * floating-point.
     * @param dst output image of the same size and depth as src.
     * @param code color space conversion code (see #ColorConversionCodes).
     * @param dstCn number of channels in the destination image; if the parameter is 0, the number of the
     * channels is derived automatically from src and code.
     *
     * SEE: REF: imgproc_color_conversions
     */
    public static void cvtColor(Mat src, Mat dst, int code, int dstCn) {
        cvtColor_0(src.nativeObj, dst.nativeObj, code, dstCn);
    }

    /**
     * Converts an image from one color space to another.
     *
     * The function converts an input image from one color space to another. In case of a transformation
     * to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note
     * that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the
     * bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue
     * component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and
     * sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.
     *
     * The conventional ranges for R, G, and B channel values are:
     * <ul>
     *   <li>
     *    0 to 255 for CV_8U images
     *   </li>
     *   <li>
     *    0 to 65535 for CV_16U images
     *   </li>
     *   <li>
     *    0 to 1 for CV_32F images
     *   </li>
     * </ul>
     *
     * In case of linear transformations, the range does not matter. But in case of a non-linear
     * transformation, an input RGB image should be normalized to the proper value range to get the correct
     * results, for example, for RGB \(\rightarrow\) L\*u\*v\* transformation. For example, if you have a
     * 32-bit floating-point image directly converted from an 8-bit image without any scaling, then it will
     * have the 0..255 value range instead of 0..1 assumed by the function. So, before calling #cvtColor ,
     * you need first to scale the image down:
     * <code>
     *     img *= 1./255;
     *     cvtColor(img, img, COLOR_BGR2Luv);
     * </code>
     * If you use #cvtColor with 8-bit images, the conversion will have some information lost. For many
     * applications, this will not be noticeable but it is recommended to use 32-bit images in applications
     * that need the full range of colors or that convert an image before an operation and then convert
     * back.
     *
     * If conversion adds the alpha channel, its value will set to the maximum of corresponding channel
     * range: 255 for CV_8U, 65535 for CV_16U, 1 for CV_32F.
     *
     * @param src input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision
     * floating-point.
     * @param dst output image of the same size and depth as src.
     * @param code color space conversion code (see #ColorConversionCodes).
     * channels is derived automatically from src and code.
     *
     * SEE: REF: imgproc_color_conversions
     */
    public static void cvtColor(Mat src, Mat dst, int code) {
        cvtColor_1(src.nativeObj, dst.nativeObj, code);
    }


    //
    // C++:  void cv::cvtColorTwoPlane(Mat src1, Mat src2, Mat& dst, int code)
    //

    /**
     * Converts an image from one color space to another where the source image is
     * stored in two planes.
     *
     * This function only supports YUV420 to RGB conversion as of now.
     *
     * <ul>
     *   <li>
     *  #COLOR_YUV2BGR_NV12
     *   </li>
     *   <li>
     *  #COLOR_YUV2RGB_NV12
     *   </li>
     *   <li>
     *  #COLOR_YUV2BGRA_NV12
     *   </li>
     *   <li>
     *  #COLOR_YUV2RGBA_NV12
     *   </li>
     *   <li>
     *  #COLOR_YUV2BGR_NV21
     *   </li>
     *   <li>
     *  #COLOR_YUV2RGB_NV21
     *   </li>
     *   <li>
     *  #COLOR_YUV2BGRA_NV21
     *   </li>
     *   <li>
     *  #COLOR_YUV2RGBA_NV21
     *   </li>
     * </ul>
     * @param src1 automatically generated
     * @param src2 automatically generated
     * @param dst automatically generated
     * @param code automatically generated
     */
    public static void cvtColorTwoPlane(Mat src1, Mat src2, Mat dst, int code) {
        cvtColorTwoPlane_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, code);
    }


    //
    // C++:  void cv::demosaicing(Mat src, Mat& dst, int code, int dstCn = 0)
    //

    /**
     * main function for all demosaicing processes
     *
     * @param src input image: 8-bit unsigned or 16-bit unsigned.
     * @param dst output image of the same size and depth as src.
     * @param code Color space conversion code (see the description below).
     * @param dstCn number of channels in the destination image; if the parameter is 0, the number of the
     * channels is derived automatically from src and code.
     *
     * The function can do the following transformations:
     *
     * <ul>
     *   <li>
     *    Demosaicing using bilinear interpolation
     *   </li>
     * </ul>
     *
     *     #COLOR_BayerBG2BGR , #COLOR_BayerGB2BGR , #COLOR_BayerRG2BGR , #COLOR_BayerGR2BGR
     *
     *     #COLOR_BayerBG2GRAY , #COLOR_BayerGB2GRAY , #COLOR_BayerRG2GRAY , #COLOR_BayerGR2GRAY
     *
     * <ul>
     *   <li>
     *    Demosaicing using Variable Number of Gradients.
     *   </li>
     * </ul>
     *
     *     #COLOR_BayerBG2BGR_VNG , #COLOR_BayerGB2BGR_VNG , #COLOR_BayerRG2BGR_VNG , #COLOR_BayerGR2BGR_VNG
     *
     * <ul>
     *   <li>
     *    Edge-Aware Demosaicing.
     *   </li>
     * </ul>
     *
     *     #COLOR_BayerBG2BGR_EA , #COLOR_BayerGB2BGR_EA , #COLOR_BayerRG2BGR_EA , #COLOR_BayerGR2BGR_EA
     *
     * <ul>
     *   <li>
     *    Demosaicing with alpha channel
     *   </li>
     * </ul>
     *
     *     #COLOR_BayerBG2BGRA , #COLOR_BayerGB2BGRA , #COLOR_BayerRG2BGRA , #COLOR_BayerGR2BGRA
     *
     * SEE: cvtColor
     */
    public static void demosaicing(Mat src, Mat dst, int code, int dstCn) {
        demosaicing_0(src.nativeObj, dst.nativeObj, code, dstCn);
    }

    /**
     * main function for all demosaicing processes
     *
     * @param src input image: 8-bit unsigned or 16-bit unsigned.
     * @param dst output image of the same size and depth as src.
     * @param code Color space conversion code (see the description below).
     * channels is derived automatically from src and code.
     *
     * The function can do the following transformations:
     *
     * <ul>
     *   <li>
     *    Demosaicing using bilinear interpolation
     *   </li>
     * </ul>
     *
     *     #COLOR_BayerBG2BGR , #COLOR_BayerGB2BGR , #COLOR_BayerRG2BGR , #COLOR_BayerGR2BGR
     *
     *     #COLOR_BayerBG2GRAY , #COLOR_BayerGB2GRAY , #COLOR_BayerRG2GRAY , #COLOR_BayerGR2GRAY
     *
     * <ul>
     *   <li>
     *    Demosaicing using Variable Number of Gradients.
     *   </li>
     * </ul>
     *
     *     #COLOR_BayerBG2BGR_VNG , #COLOR_BayerGB2BGR_VNG , #COLOR_BayerRG2BGR_VNG , #COLOR_BayerGR2BGR_VNG
     *
     * <ul>
     *   <li>
     *    Edge-Aware Demosaicing.
     *   </li>
     * </ul>
     *
     *     #COLOR_BayerBG2BGR_EA , #COLOR_BayerGB2BGR_EA , #COLOR_BayerRG2BGR_EA , #COLOR_BayerGR2BGR_EA
     *
     * <ul>
     *   <li>
     *    Demosaicing with alpha channel
     *   </li>
     * </ul>
     *
     *     #COLOR_BayerBG2BGRA , #COLOR_BayerGB2BGRA , #COLOR_BayerRG2BGRA , #COLOR_BayerGR2BGRA
     *
     * SEE: cvtColor
     */
    public static void demosaicing(Mat src, Mat dst, int code) {
        demosaicing_1(src.nativeObj, dst.nativeObj, code);
    }


    //
    // C++:  void cv::dilate(Mat src, Mat& dst, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
    //

    /**
     * Dilates an image by using a specific structuring element.
     *
     * The function dilates the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the maximum is taken:
     * \(\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement
     * @param anchor position of the anchor within the element; default value (-1, -1) means that the
     * anchor is at the element center.
     * @param iterations number of times dilation is applied.
     * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not suported.
     * @param borderValue border value in case of a constant border
     * SEE:  erode, morphologyEx, getStructuringElement
     */
    public static void dilate(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType, Scalar borderValue) {
        dilate_0(src.nativeObj, dst.nativeObj, kernel.nativeObj, anchor.x, anchor.y, iterations, borderType, borderValue.val[0], borderValue.val[1], borderValue.val[2], borderValue.val[3]);
    }

    /**
     * Dilates an image by using a specific structuring element.
     *
     * The function dilates the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the maximum is taken:
     * \(\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement
     * @param anchor position of the anchor within the element; default value (-1, -1) means that the
     * anchor is at the element center.
     * @param iterations number of times dilation is applied.
     * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not suported.
     * SEE:  erode, morphologyEx, getStructuringElement
     */
    public static void dilate(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType) {
        dilate_1(src.nativeObj, dst.nativeObj, kernel.nativeObj, anchor.x, anchor.y, iterations, borderType);
    }

    /**
     * Dilates an image by using a specific structuring element.
     *
     * The function dilates the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the maximum is taken:
     * \(\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement
     * @param anchor position of the anchor within the element; default value (-1, -1) means that the
     * anchor is at the element center.
     * @param iterations number of times dilation is applied.
     * SEE:  erode, morphologyEx, getStructuringElement
     */
    public static void dilate(Mat src, Mat dst, Mat kernel, Point anchor, int iterations) {
        dilate_2(src.nativeObj, dst.nativeObj, kernel.nativeObj, anchor.x, anchor.y, iterations);
    }

    /**
     * Dilates an image by using a specific structuring element.
     *
     * The function dilates the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the maximum is taken:
     * \(\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement
     * @param anchor position of the anchor within the element; default value (-1, -1) means that the
     * anchor is at the element center.
     * SEE:  erode, morphologyEx, getStructuringElement
     */
    public static void dilate(Mat src, Mat dst, Mat kernel, Point anchor) {
        dilate_3(src.nativeObj, dst.nativeObj, kernel.nativeObj, anchor.x, anchor.y);
    }

    /**
     * Dilates an image by using a specific structuring element.
     *
     * The function dilates the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the maximum is taken:
     * \(\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement
     * anchor is at the element center.
     * SEE:  erode, morphologyEx, getStructuringElement
     */
    public static void dilate(Mat src, Mat dst, Mat kernel) {
        dilate_4(src.nativeObj, dst.nativeObj, kernel.nativeObj);
    }


    //
    // C++:  void cv::distanceTransform(Mat src, Mat& dst, Mat& labels, int distanceType, int maskSize, int labelType = DIST_LABEL_CCOMP)
    //

    /**
     * Calculates the distance to the closest zero pixel for each pixel of the source image.
     *
     * The function cv::distanceTransform calculates the approximate or precise distance from every binary
     * image pixel to the nearest zero pixel. For zero image pixels, the distance will obviously be zero.
     *
     * When maskSize == #DIST_MASK_PRECISE and distanceType == #DIST_L2 , the function runs the
     * algorithm described in CITE: Felzenszwalb04 . This algorithm is parallelized with the TBB library.
     *
     * In other cases, the algorithm CITE: Borgefors86 is used. This means that for a pixel the function
     * finds the shortest path to the nearest zero pixel consisting of basic shifts: horizontal, vertical,
     * diagonal, or knight's move (the latest is available for a \(5\times 5\) mask). The overall
     * distance is calculated as a sum of these basic distances. Since the distance function should be
     * symmetric, all of the horizontal and vertical shifts must have the same cost (denoted as a ), all
     * the diagonal shifts must have the same cost (denoted as {@code b}), and all knight's moves must have the
     * same cost (denoted as {@code c}). For the #DIST_C and #DIST_L1 types, the distance is calculated
     * precisely, whereas for #DIST_L2 (Euclidean distance) the distance can be calculated only with a
     * relative error (a \(5\times 5\) mask gives more accurate results). For {@code a},{@code b}, and {@code c}, OpenCV
     * uses the values suggested in the original paper:
     * <ul>
     *   <li>
     *  DIST_L1: {@code a = 1, b = 2}
     *   </li>
     *   <li>
     *  DIST_L2:
     *   <ul>
     *     <li>
     *      {@code 3 x 3}: {@code a=0.955, b=1.3693}
     *     </li>
     *     <li>
     *      {@code 5 x 5}: {@code a=1, b=1.4, c=2.1969}
     *     </li>
     *   </ul>
     *   <li>
     *  DIST_C: {@code a = 1, b = 1}
     *   </li>
     * </ul>
     *
     * Typically, for a fast, coarse distance estimation #DIST_L2, a \(3\times 3\) mask is used. For a
     * more accurate distance estimation #DIST_L2, a \(5\times 5\) mask or the precise algorithm is used.
     * Note that both the precise and the approximate algorithms are linear on the number of pixels.
     *
     * This variant of the function does not only compute the minimum distance for each pixel \((x, y)\)
     * but also identifies the nearest connected component consisting of zero pixels
     * (labelType==#DIST_LABEL_CCOMP) or the nearest zero pixel (labelType==#DIST_LABEL_PIXEL). Index of the
     * component/pixel is stored in {@code labels(x, y)}. When labelType==#DIST_LABEL_CCOMP, the function
     * automatically finds connected components of zero pixels in the input image and marks them with
     * distinct labels. When labelType==#DIST_LABEL_CCOMP, the function scans through the input image and
     * marks all the zero pixels with distinct labels.
     *
     * In this mode, the complexity is still linear. That is, the function provides a very fast way to
     * compute the Voronoi diagram for a binary image. Currently, the second variant can use only the
     * approximate distance transform algorithm, i.e. maskSize=#DIST_MASK_PRECISE is not supported
     * yet.
     *
     * @param src 8-bit, single-channel (binary) source image.
     * @param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,
     * single-channel image of the same size as src.
     * @param labels Output 2D array of labels (the discrete Voronoi diagram). It has the type
     * CV_32SC1 and the same size as src.
     * @param distanceType Type of distance, see #DistanceTypes
     * @param maskSize Size of the distance transform mask, see #DistanceTransformMasks.
     * #DIST_MASK_PRECISE is not supported by this variant. In case of the #DIST_L1 or #DIST_C distance type,
     * the parameter is forced to 3 because a \(3\times 3\) mask gives the same result as \(5\times
     * 5\) or any larger aperture.
     * @param labelType Type of the label array to build, see #DistanceTransformLabelTypes.
     */
    public static void distanceTransformWithLabels(Mat src, Mat dst, Mat labels, int distanceType, int maskSize, int labelType) {
        distanceTransformWithLabels_0(src.nativeObj, dst.nativeObj, labels.nativeObj, distanceType, maskSize, labelType);
    }

    /**
     * Calculates the distance to the closest zero pixel for each pixel of the source image.
     *
     * The function cv::distanceTransform calculates the approximate or precise distance from every binary
     * image pixel to the nearest zero pixel. For zero image pixels, the distance will obviously be zero.
     *
     * When maskSize == #DIST_MASK_PRECISE and distanceType == #DIST_L2 , the function runs the
     * algorithm described in CITE: Felzenszwalb04 . This algorithm is parallelized with the TBB library.
     *
     * In other cases, the algorithm CITE: Borgefors86 is used. This means that for a pixel the function
     * finds the shortest path to the nearest zero pixel consisting of basic shifts: horizontal, vertical,
     * diagonal, or knight's move (the latest is available for a \(5\times 5\) mask). The overall
     * distance is calculated as a sum of these basic distances. Since the distance function should be
     * symmetric, all of the horizontal and vertical shifts must have the same cost (denoted as a ), all
     * the diagonal shifts must have the same cost (denoted as {@code b}), and all knight's moves must have the
     * same cost (denoted as {@code c}). For the #DIST_C and #DIST_L1 types, the distance is calculated
     * precisely, whereas for #DIST_L2 (Euclidean distance) the distance can be calculated only with a
     * relative error (a \(5\times 5\) mask gives more accurate results). For {@code a},{@code b}, and {@code c}, OpenCV
     * uses the values suggested in the original paper:
     * <ul>
     *   <li>
     *  DIST_L1: {@code a = 1, b = 2}
     *   </li>
     *   <li>
     *  DIST_L2:
     *   <ul>
     *     <li>
     *      {@code 3 x 3}: {@code a=0.955, b=1.3693}
     *     </li>
     *     <li>
     *      {@code 5 x 5}: {@code a=1, b=1.4, c=2.1969}
     *     </li>
     *   </ul>
     *   <li>
     *  DIST_C: {@code a = 1, b = 1}
     *   </li>
     * </ul>
     *
     * Typically, for a fast, coarse distance estimation #DIST_L2, a \(3\times 3\) mask is used. For a
     * more accurate distance estimation #DIST_L2, a \(5\times 5\) mask or the precise algorithm is used.
     * Note that both the precise and the approximate algorithms are linear on the number of pixels.
     *
     * This variant of the function does not only compute the minimum distance for each pixel \((x, y)\)
     * but also identifies the nearest connected component consisting of zero pixels
     * (labelType==#DIST_LABEL_CCOMP) or the nearest zero pixel (labelType==#DIST_LABEL_PIXEL). Index of the
     * component/pixel is stored in {@code labels(x, y)}. When labelType==#DIST_LABEL_CCOMP, the function
     * automatically finds connected components of zero pixels in the input image and marks them with
     * distinct labels. When labelType==#DIST_LABEL_CCOMP, the function scans through the input image and
     * marks all the zero pixels with distinct labels.
     *
     * In this mode, the complexity is still linear. That is, the function provides a very fast way to
     * compute the Voronoi diagram for a binary image. Currently, the second variant can use only the
     * approximate distance transform algorithm, i.e. maskSize=#DIST_MASK_PRECISE is not supported
     * yet.
     *
     * @param src 8-bit, single-channel (binary) source image.
     * @param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,
     * single-channel image of the same size as src.
     * @param labels Output 2D array of labels (the discrete Voronoi diagram). It has the type
     * CV_32SC1 and the same size as src.
     * @param distanceType Type of distance, see #DistanceTypes
     * @param maskSize Size of the distance transform mask, see #DistanceTransformMasks.
     * #DIST_MASK_PRECISE is not supported by this variant. In case of the #DIST_L1 or #DIST_C distance type,
     * the parameter is forced to 3 because a \(3\times 3\) mask gives the same result as \(5\times
     * 5\) or any larger aperture.
     */
    public static void distanceTransformWithLabels(Mat src, Mat dst, Mat labels, int distanceType, int maskSize) {
        distanceTransformWithLabels_1(src.nativeObj, dst.nativeObj, labels.nativeObj, distanceType, maskSize);
    }


    //
    // C++:  void cv::distanceTransform(Mat src, Mat& dst, int distanceType, int maskSize, int dstType = CV_32F)
    //

    /**
     *
     * @param src 8-bit, single-channel (binary) source image.
     * @param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,
     * single-channel image of the same size as src .
     * @param distanceType Type of distance, see #DistanceTypes
     * @param maskSize Size of the distance transform mask, see #DistanceTransformMasks. In case of the
     * #DIST_L1 or #DIST_C distance type, the parameter is forced to 3 because a \(3\times 3\) mask gives
     * the same result as \(5\times 5\) or any larger aperture.
     * @param dstType Type of output image. It can be CV_8U or CV_32F. Type CV_8U can be used only for
     * the first variant of the function and distanceType == #DIST_L1.
     */
    public static void distanceTransform(Mat src, Mat dst, int distanceType, int maskSize, int dstType) {
        distanceTransform_0(src.nativeObj, dst.nativeObj, distanceType, maskSize, dstType);
    }

    /**
     *
     * @param src 8-bit, single-channel (binary) source image.
     * @param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,
     * single-channel image of the same size as src .
     * @param distanceType Type of distance, see #DistanceTypes
     * @param maskSize Size of the distance transform mask, see #DistanceTransformMasks. In case of the
     * #DIST_L1 or #DIST_C distance type, the parameter is forced to 3 because a \(3\times 3\) mask gives
     * the same result as \(5\times 5\) or any larger aperture.
     * the first variant of the function and distanceType == #DIST_L1.
     */
    public static void distanceTransform(Mat src, Mat dst, int distanceType, int maskSize) {
        distanceTransform_1(src.nativeObj, dst.nativeObj, distanceType, maskSize);
    }


    //
    // C++:  void cv::drawContours(Mat& image, vector_vector_Point contours, int contourIdx, Scalar color, int thickness = 1, int lineType = LINE_8, Mat hierarchy = Mat(), int maxLevel = INT_MAX, Point offset = Point())
    //

    /**
     * Draws contours outlines or filled contours.
     *
     * The function draws contour outlines in the image if \(\texttt{thickness} \ge 0\) or fills the area
     * bounded by the contours if \(\texttt{thickness}&lt;0\) . The example below shows how to retrieve
     * connected components from the binary image and label them: :
     * INCLUDE: snippets/imgproc_drawContours.cpp
     *
     * @param image Destination image.
     * @param contours All the input contours. Each contour is stored as a point vector.
     * @param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
     * @param color Color of the contours.
     * @param thickness Thickness of lines the contours are drawn with. If it is negative (for example,
     * thickness=#FILLED ), the contour interiors are drawn.
     * @param lineType Line connectivity. See #LineTypes
     * @param hierarchy Optional information about hierarchy. It is only needed if you want to draw only
     * some of the contours (see maxLevel ).
     * @param maxLevel Maximal level for drawn contours. If it is 0, only the specified contour is drawn.
     * If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
     * draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
     * parameter is only taken into account when there is hierarchy available.
     * @param offset Optional contour shift parameter. Shift all the drawn contours by the specified
     * \(\texttt{offset}=(dx,dy)\) .
     * <b>Note:</b> When thickness=#FILLED, the function is designed to handle connected components with holes correctly
     * even when no hierarchy date is provided. This is done by analyzing all the outlines together
     * using even-odd rule. This may give incorrect results if you have a joint collection of separately retrieved
     * contours. In order to solve this problem, you need to call #drawContours separately for each sub-group
     * of contours, or iterate over the collection using contourIdx parameter.
     */
    public static void drawContours(Mat image, List<MatOfPoint> contours, int contourIdx, Scalar color, int thickness, int lineType, Mat hierarchy, int maxLevel, Point offset) {
        List<Mat> contours_tmplm = new ArrayList<Mat>((contours != null) ? contours.size() : 0);
        Mat contours_mat = Converters.vector_vector_Point_to_Mat(contours, contours_tmplm);
        drawContours_0(image.nativeObj, contours_mat.nativeObj, contourIdx, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, hierarchy.nativeObj, maxLevel, offset.x, offset.y);
    }

    /**
     * Draws contours outlines or filled contours.
     *
     * The function draws contour outlines in the image if \(\texttt{thickness} \ge 0\) or fills the area
     * bounded by the contours if \(\texttt{thickness}&lt;0\) . The example below shows how to retrieve
     * connected components from the binary image and label them: :
     * INCLUDE: snippets/imgproc_drawContours.cpp
     *
     * @param image Destination image.
     * @param contours All the input contours. Each contour is stored as a point vector.
     * @param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
     * @param color Color of the contours.
     * @param thickness Thickness of lines the contours are drawn with. If it is negative (for example,
     * thickness=#FILLED ), the contour interiors are drawn.
     * @param lineType Line connectivity. See #LineTypes
     * @param hierarchy Optional information about hierarchy. It is only needed if you want to draw only
     * some of the contours (see maxLevel ).
     * @param maxLevel Maximal level for drawn contours. If it is 0, only the specified contour is drawn.
     * If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
     * draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
     * parameter is only taken into account when there is hierarchy available.
     * \(\texttt{offset}=(dx,dy)\) .
     * <b>Note:</b> When thickness=#FILLED, the function is designed to handle connected components with holes correctly
     * even when no hierarchy date is provided. This is done by analyzing all the outlines together
     * using even-odd rule. This may give incorrect results if you have a joint collection of separately retrieved
     * contours. In order to solve this problem, you need to call #drawContours separately for each sub-group
     * of contours, or iterate over the collection using contourIdx parameter.
     */
    public static void drawContours(Mat image, List<MatOfPoint> contours, int contourIdx, Scalar color, int thickness, int lineType, Mat hierarchy, int maxLevel) {
        List<Mat> contours_tmplm = new ArrayList<Mat>((contours != null) ? contours.size() : 0);
        Mat contours_mat = Converters.vector_vector_Point_to_Mat(contours, contours_tmplm);
        drawContours_1(image.nativeObj, contours_mat.nativeObj, contourIdx, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, hierarchy.nativeObj, maxLevel);
    }

    /**
     * Draws contours outlines or filled contours.
     *
     * The function draws contour outlines in the image if \(\texttt{thickness} \ge 0\) or fills the area
     * bounded by the contours if \(\texttt{thickness}&lt;0\) . The example below shows how to retrieve
     * connected components from the binary image and label them: :
     * INCLUDE: snippets/imgproc_drawContours.cpp
     *
     * @param image Destination image.
     * @param contours All the input contours. Each contour is stored as a point vector.
     * @param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
     * @param color Color of the contours.
     * @param thickness Thickness of lines the contours are drawn with. If it is negative (for example,
     * thickness=#FILLED ), the contour interiors are drawn.
     * @param lineType Line connectivity. See #LineTypes
     * @param hierarchy Optional information about hierarchy. It is only needed if you want to draw only
     * some of the contours (see maxLevel ).
     * If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
     * draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
     * parameter is only taken into account when there is hierarchy available.
     * \(\texttt{offset}=(dx,dy)\) .
     * <b>Note:</b> When thickness=#FILLED, the function is designed to handle connected components with holes correctly
     * even when no hierarchy date is provided. This is done by analyzing all the outlines together
     * using even-odd rule. This may give incorrect results if you have a joint collection of separately retrieved
     * contours. In order to solve this problem, you need to call #drawContours separately for each sub-group
     * of contours, or iterate over the collection using contourIdx parameter.
     */
    public static void drawContours(Mat image, List<MatOfPoint> contours, int contourIdx, Scalar color, int thickness, int lineType, Mat hierarchy) {
        List<Mat> contours_tmplm = new ArrayList<Mat>((contours != null) ? contours.size() : 0);
        Mat contours_mat = Converters.vector_vector_Point_to_Mat(contours, contours_tmplm);
        drawContours_2(image.nativeObj, contours_mat.nativeObj, contourIdx, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, hierarchy.nativeObj);
    }

    /**
     * Draws contours outlines or filled contours.
     *
     * The function draws contour outlines in the image if \(\texttt{thickness} \ge 0\) or fills the area
     * bounded by the contours if \(\texttt{thickness}&lt;0\) . The example below shows how to retrieve
     * connected components from the binary image and label them: :
     * INCLUDE: snippets/imgproc_drawContours.cpp
     *
     * @param image Destination image.
     * @param contours All the input contours. Each contour is stored as a point vector.
     * @param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
     * @param color Color of the contours.
     * @param thickness Thickness of lines the contours are drawn with. If it is negative (for example,
     * thickness=#FILLED ), the contour interiors are drawn.
     * @param lineType Line connectivity. See #LineTypes
     * some of the contours (see maxLevel ).
     * If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
     * draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
     * parameter is only taken into account when there is hierarchy available.
     * \(\texttt{offset}=(dx,dy)\) .
     * <b>Note:</b> When thickness=#FILLED, the function is designed to handle connected components with holes correctly
     * even when no hierarchy date is provided. This is done by analyzing all the outlines together
     * using even-odd rule. This may give incorrect results if you have a joint collection of separately retrieved
     * contours. In order to solve this problem, you need to call #drawContours separately for each sub-group
     * of contours, or iterate over the collection using contourIdx parameter.
     */
    public static void drawContours(Mat image, List<MatOfPoint> contours, int contourIdx, Scalar color, int thickness, int lineType) {
        List<Mat> contours_tmplm = new ArrayList<Mat>((contours != null) ? contours.size() : 0);
        Mat contours_mat = Converters.vector_vector_Point_to_Mat(contours, contours_tmplm);
        drawContours_3(image.nativeObj, contours_mat.nativeObj, contourIdx, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     * Draws contours outlines or filled contours.
     *
     * The function draws contour outlines in the image if \(\texttt{thickness} \ge 0\) or fills the area
     * bounded by the contours if \(\texttt{thickness}&lt;0\) . The example below shows how to retrieve
     * connected components from the binary image and label them: :
     * INCLUDE: snippets/imgproc_drawContours.cpp
     *
     * @param image Destination image.
     * @param contours All the input contours. Each contour is stored as a point vector.
     * @param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
     * @param color Color of the contours.
     * @param thickness Thickness of lines the contours are drawn with. If it is negative (for example,
     * thickness=#FILLED ), the contour interiors are drawn.
     * some of the contours (see maxLevel ).
     * If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
     * draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
     * parameter is only taken into account when there is hierarchy available.
     * \(\texttt{offset}=(dx,dy)\) .
     * <b>Note:</b> When thickness=#FILLED, the function is designed to handle connected components with holes correctly
     * even when no hierarchy date is provided. This is done by analyzing all the outlines together
     * using even-odd rule. This may give incorrect results if you have a joint collection of separately retrieved
     * contours. In order to solve this problem, you need to call #drawContours separately for each sub-group
     * of contours, or iterate over the collection using contourIdx parameter.
     */
    public static void drawContours(Mat image, List<MatOfPoint> contours, int contourIdx, Scalar color, int thickness) {
        List<Mat> contours_tmplm = new ArrayList<Mat>((contours != null) ? contours.size() : 0);
        Mat contours_mat = Converters.vector_vector_Point_to_Mat(contours, contours_tmplm);
        drawContours_4(image.nativeObj, contours_mat.nativeObj, contourIdx, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     * Draws contours outlines or filled contours.
     *
     * The function draws contour outlines in the image if \(\texttt{thickness} \ge 0\) or fills the area
     * bounded by the contours if \(\texttt{thickness}&lt;0\) . The example below shows how to retrieve
     * connected components from the binary image and label them: :
     * INCLUDE: snippets/imgproc_drawContours.cpp
     *
     * @param image Destination image.
     * @param contours All the input contours. Each contour is stored as a point vector.
     * @param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
     * @param color Color of the contours.
     * thickness=#FILLED ), the contour interiors are drawn.
     * some of the contours (see maxLevel ).
     * If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
     * draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
     * parameter is only taken into account when there is hierarchy available.
     * \(\texttt{offset}=(dx,dy)\) .
     * <b>Note:</b> When thickness=#FILLED, the function is designed to handle connected components with holes correctly
     * even when no hierarchy date is provided. This is done by analyzing all the outlines together
     * using even-odd rule. This may give incorrect results if you have a joint collection of separately retrieved
     * contours. In order to solve this problem, you need to call #drawContours separately for each sub-group
     * of contours, or iterate over the collection using contourIdx parameter.
     */
    public static void drawContours(Mat image, List<MatOfPoint> contours, int contourIdx, Scalar color) {
        List<Mat> contours_tmplm = new ArrayList<Mat>((contours != null) ? contours.size() : 0);
        Mat contours_mat = Converters.vector_vector_Point_to_Mat(contours, contours_tmplm);
        drawContours_5(image.nativeObj, contours_mat.nativeObj, contourIdx, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::drawMarker(Mat& img, Point position, Scalar color, int markerType = MARKER_CROSS, int markerSize = 20, int thickness = 1, int line_type = 8)
    //

    /**
     * Draws a marker on a predefined position in an image.
     *
     * The function cv::drawMarker draws a marker on a given position in the image. For the moment several
     * marker types are supported, see #MarkerTypes for more information.
     *
     * @param img Image.
     * @param position The point where the crosshair is positioned.
     * @param color Line color.
     * @param markerType The specific type of marker you want to use, see #MarkerTypes
     * @param thickness Line thickness.
     * @param line_type Type of the line, See #LineTypes
     * @param markerSize The length of the marker axis [default = 20 pixels]
     */
    public static void drawMarker(Mat img, Point position, Scalar color, int markerType, int markerSize, int thickness, int line_type) {
        drawMarker_0(img.nativeObj, position.x, position.y, color.val[0], color.val[1], color.val[2], color.val[3], markerType, markerSize, thickness, line_type);
    }

    /**
     * Draws a marker on a predefined position in an image.
     *
     * The function cv::drawMarker draws a marker on a given position in the image. For the moment several
     * marker types are supported, see #MarkerTypes for more information.
     *
     * @param img Image.
     * @param position The point where the crosshair is positioned.
     * @param color Line color.
     * @param markerType The specific type of marker you want to use, see #MarkerTypes
     * @param thickness Line thickness.
     * @param markerSize The length of the marker axis [default = 20 pixels]
     */
    public static void drawMarker(Mat img, Point position, Scalar color, int markerType, int markerSize, int thickness) {
        drawMarker_1(img.nativeObj, position.x, position.y, color.val[0], color.val[1], color.val[2], color.val[3], markerType, markerSize, thickness);
    }

    /**
     * Draws a marker on a predefined position in an image.
     *
     * The function cv::drawMarker draws a marker on a given position in the image. For the moment several
     * marker types are supported, see #MarkerTypes for more information.
     *
     * @param img Image.
     * @param position The point where the crosshair is positioned.
     * @param color Line color.
     * @param markerType The specific type of marker you want to use, see #MarkerTypes
     * @param markerSize The length of the marker axis [default = 20 pixels]
     */
    public static void drawMarker(Mat img, Point position, Scalar color, int markerType, int markerSize) {
        drawMarker_2(img.nativeObj, position.x, position.y, color.val[0], color.val[1], color.val[2], color.val[3], markerType, markerSize);
    }

    /**
     * Draws a marker on a predefined position in an image.
     *
     * The function cv::drawMarker draws a marker on a given position in the image. For the moment several
     * marker types are supported, see #MarkerTypes for more information.
     *
     * @param img Image.
     * @param position The point where the crosshair is positioned.
     * @param color Line color.
     * @param markerType The specific type of marker you want to use, see #MarkerTypes
     */
    public static void drawMarker(Mat img, Point position, Scalar color, int markerType) {
        drawMarker_3(img.nativeObj, position.x, position.y, color.val[0], color.val[1], color.val[2], color.val[3], markerType);
    }

    /**
     * Draws a marker on a predefined position in an image.
     *
     * The function cv::drawMarker draws a marker on a given position in the image. For the moment several
     * marker types are supported, see #MarkerTypes for more information.
     *
     * @param img Image.
     * @param position The point where the crosshair is positioned.
     * @param color Line color.
     */
    public static void drawMarker(Mat img, Point position, Scalar color) {
        drawMarker_4(img.nativeObj, position.x, position.y, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::ellipse(Mat& img, Point center, Size axes, double angle, double startAngle, double endAngle, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    //

    /**
     * Draws a simple or thick elliptic arc or fills an ellipse sector.
     *
     * The function cv::ellipse with more parameters draws an ellipse outline, a filled ellipse, an elliptic
     * arc, or a filled ellipse sector. The drawing code uses general parametric form.
     * A piecewise-linear curve is used to approximate the elliptic arc
     * boundary. If you need more control of the ellipse rendering, you can retrieve the curve using
     * #ellipse2Poly and then render it with #polylines or fill it with #fillPoly. If you use the first
     * variant of the function and want to draw the whole ellipse, not an arc, pass {@code startAngle=0} and
     * {@code endAngle=360}. If {@code startAngle} is greater than {@code endAngle}, they are swapped. The figure below explains
     * the meaning of the parameters to draw the blue arc.
     *
     * ![Parameters of Elliptic Arc](pics/ellipse.svg)
     *
     * @param img Image.
     * @param center Center of the ellipse.
     * @param axes Half of the size of the ellipse main axes.
     * @param angle Ellipse rotation angle in degrees.
     * @param startAngle Starting angle of the elliptic arc in degrees.
     * @param endAngle Ending angle of the elliptic arc in degrees.
     * @param color Ellipse color.
     * @param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that
     * a filled ellipse sector is to be drawn.
     * @param lineType Type of the ellipse boundary. See #LineTypes
     * @param shift Number of fractional bits in the coordinates of the center and values of axes.
     */
    public static void ellipse(Mat img, Point center, Size axes, double angle, double startAngle, double endAngle, Scalar color, int thickness, int lineType, int shift) {
        ellipse_0(img.nativeObj, center.x, center.y, axes.width, axes.height, angle, startAngle, endAngle, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, shift);
    }

    /**
     * Draws a simple or thick elliptic arc or fills an ellipse sector.
     *
     * The function cv::ellipse with more parameters draws an ellipse outline, a filled ellipse, an elliptic
     * arc, or a filled ellipse sector. The drawing code uses general parametric form.
     * A piecewise-linear curve is used to approximate the elliptic arc
     * boundary. If you need more control of the ellipse rendering, you can retrieve the curve using
     * #ellipse2Poly and then render it with #polylines or fill it with #fillPoly. If you use the first
     * variant of the function and want to draw the whole ellipse, not an arc, pass {@code startAngle=0} and
     * {@code endAngle=360}. If {@code startAngle} is greater than {@code endAngle}, they are swapped. The figure below explains
     * the meaning of the parameters to draw the blue arc.
     *
     * ![Parameters of Elliptic Arc](pics/ellipse.svg)
     *
     * @param img Image.
     * @param center Center of the ellipse.
     * @param axes Half of the size of the ellipse main axes.
     * @param angle Ellipse rotation angle in degrees.
     * @param startAngle Starting angle of the elliptic arc in degrees.
     * @param endAngle Ending angle of the elliptic arc in degrees.
     * @param color Ellipse color.
     * @param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that
     * a filled ellipse sector is to be drawn.
     * @param lineType Type of the ellipse boundary. See #LineTypes
     */
    public static void ellipse(Mat img, Point center, Size axes, double angle, double startAngle, double endAngle, Scalar color, int thickness, int lineType) {
        ellipse_1(img.nativeObj, center.x, center.y, axes.width, axes.height, angle, startAngle, endAngle, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     * Draws a simple or thick elliptic arc or fills an ellipse sector.
     *
     * The function cv::ellipse with more parameters draws an ellipse outline, a filled ellipse, an elliptic
     * arc, or a filled ellipse sector. The drawing code uses general parametric form.
     * A piecewise-linear curve is used to approximate the elliptic arc
     * boundary. If you need more control of the ellipse rendering, you can retrieve the curve using
     * #ellipse2Poly and then render it with #polylines or fill it with #fillPoly. If you use the first
     * variant of the function and want to draw the whole ellipse, not an arc, pass {@code startAngle=0} and
     * {@code endAngle=360}. If {@code startAngle} is greater than {@code endAngle}, they are swapped. The figure below explains
     * the meaning of the parameters to draw the blue arc.
     *
     * ![Parameters of Elliptic Arc](pics/ellipse.svg)
     *
     * @param img Image.
     * @param center Center of the ellipse.
     * @param axes Half of the size of the ellipse main axes.
     * @param angle Ellipse rotation angle in degrees.
     * @param startAngle Starting angle of the elliptic arc in degrees.
     * @param endAngle Ending angle of the elliptic arc in degrees.
     * @param color Ellipse color.
     * @param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that
     * a filled ellipse sector is to be drawn.
     */
    public static void ellipse(Mat img, Point center, Size axes, double angle, double startAngle, double endAngle, Scalar color, int thickness) {
        ellipse_2(img.nativeObj, center.x, center.y, axes.width, axes.height, angle, startAngle, endAngle, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     * Draws a simple or thick elliptic arc or fills an ellipse sector.
     *
     * The function cv::ellipse with more parameters draws an ellipse outline, a filled ellipse, an elliptic
     * arc, or a filled ellipse sector. The drawing code uses general parametric form.
     * A piecewise-linear curve is used to approximate the elliptic arc
     * boundary. If you need more control of the ellipse rendering, you can retrieve the curve using
     * #ellipse2Poly and then render it with #polylines or fill it with #fillPoly. If you use the first
     * variant of the function and want to draw the whole ellipse, not an arc, pass {@code startAngle=0} and
     * {@code endAngle=360}. If {@code startAngle} is greater than {@code endAngle}, they are swapped. The figure below explains
     * the meaning of the parameters to draw the blue arc.
     *
     * ![Parameters of Elliptic Arc](pics/ellipse.svg)
     *
     * @param img Image.
     * @param center Center of the ellipse.
     * @param axes Half of the size of the ellipse main axes.
     * @param angle Ellipse rotation angle in degrees.
     * @param startAngle Starting angle of the elliptic arc in degrees.
     * @param endAngle Ending angle of the elliptic arc in degrees.
     * @param color Ellipse color.
     * a filled ellipse sector is to be drawn.
     */
    public static void ellipse(Mat img, Point center, Size axes, double angle, double startAngle, double endAngle, Scalar color) {
        ellipse_3(img.nativeObj, center.x, center.y, axes.width, axes.height, angle, startAngle, endAngle, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::ellipse(Mat& img, RotatedRect box, Scalar color, int thickness = 1, int lineType = LINE_8)
    //

    /**
     *
     * @param img Image.
     * @param box Alternative ellipse representation via RotatedRect. This means that the function draws
     * an ellipse inscribed in the rotated rectangle.
     * @param color Ellipse color.
     * @param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that
     * a filled ellipse sector is to be drawn.
     * @param lineType Type of the ellipse boundary. See #LineTypes
     */
    public static void ellipse(Mat img, RotatedRect box, Scalar color, int thickness, int lineType) {
        ellipse_4(img.nativeObj, box.center.x, box.center.y, box.size.width, box.size.height, box.angle, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     *
     * @param img Image.
     * @param box Alternative ellipse representation via RotatedRect. This means that the function draws
     * an ellipse inscribed in the rotated rectangle.
     * @param color Ellipse color.
     * @param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that
     * a filled ellipse sector is to be drawn.
     */
    public static void ellipse(Mat img, RotatedRect box, Scalar color, int thickness) {
        ellipse_5(img.nativeObj, box.center.x, box.center.y, box.size.width, box.size.height, box.angle, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     *
     * @param img Image.
     * @param box Alternative ellipse representation via RotatedRect. This means that the function draws
     * an ellipse inscribed in the rotated rectangle.
     * @param color Ellipse color.
     * a filled ellipse sector is to be drawn.
     */
    public static void ellipse(Mat img, RotatedRect box, Scalar color) {
        ellipse_6(img.nativeObj, box.center.x, box.center.y, box.size.width, box.size.height, box.angle, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::ellipse2Poly(Point center, Size axes, int angle, int arcStart, int arcEnd, int delta, vector_Point& pts)
    //

    /**
     * Approximates an elliptic arc with a polyline.
     *
     * The function ellipse2Poly computes the vertices of a polyline that approximates the specified
     * elliptic arc. It is used by #ellipse. If {@code arcStart} is greater than {@code arcEnd}, they are swapped.
     *
     * @param center Center of the arc.
     * @param axes Half of the size of the ellipse main axes. See #ellipse for details.
     * @param angle Rotation angle of the ellipse in degrees. See #ellipse for details.
     * @param arcStart Starting angle of the elliptic arc in degrees.
     * @param arcEnd Ending angle of the elliptic arc in degrees.
     * @param delta Angle between the subsequent polyline vertices. It defines the approximation
     * accuracy.
     * @param pts Output vector of polyline vertices.
     */
    public static void ellipse2Poly(Point center, Size axes, int angle, int arcStart, int arcEnd, int delta, MatOfPoint pts) {
        Mat pts_mat = pts;
        ellipse2Poly_0(center.x, center.y, axes.width, axes.height, angle, arcStart, arcEnd, delta, pts_mat.nativeObj);
    }


    //
    // C++:  void cv::equalizeHist(Mat src, Mat& dst)
    //

    /**
     * Equalizes the histogram of a grayscale image.
     *
     * The function equalizes the histogram of the input image using the following algorithm:
     *
     * <ul>
     *   <li>
     *  Calculate the histogram \(H\) for src .
     *   </li>
     *   <li>
     *  Normalize the histogram so that the sum of histogram bins is 255.
     *   </li>
     *   <li>
     *  Compute the integral of the histogram:
     * \(H'_i =  \sum _{0  \le j &lt; i} H(j)\)
     *   </li>
     *   <li>
     *  Transform the image using \(H'\) as a look-up table: \(\texttt{dst}(x,y) = H'(\texttt{src}(x,y))\)
     *   </li>
     * </ul>
     *
     * The algorithm normalizes the brightness and increases the contrast of the image.
     *
     * @param src Source 8-bit single channel image.
     * @param dst Destination image of the same size and type as src .
     */
    public static void equalizeHist(Mat src, Mat dst) {
        equalizeHist_0(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::erode(Mat src, Mat& dst, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
    //

    /**
     * Erodes an image by using a specific structuring element.
     *
     * The function erodes the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the minimum is taken:
     *
     * \(\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for erosion; if {@code element=Mat()}, a {@code 3 x 3} rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement.
     * @param anchor position of the anchor within the element; default value (-1, -1) means that the
     * anchor is at the element center.
     * @param iterations number of times erosion is applied.
     * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * @param borderValue border value in case of a constant border
     * SEE:  dilate, morphologyEx, getStructuringElement
     */
    public static void erode(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType, Scalar borderValue) {
        erode_0(src.nativeObj, dst.nativeObj, kernel.nativeObj, anchor.x, anchor.y, iterations, borderType, borderValue.val[0], borderValue.val[1], borderValue.val[2], borderValue.val[3]);
    }

    /**
     * Erodes an image by using a specific structuring element.
     *
     * The function erodes the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the minimum is taken:
     *
     * \(\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for erosion; if {@code element=Mat()}, a {@code 3 x 3} rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement.
     * @param anchor position of the anchor within the element; default value (-1, -1) means that the
     * anchor is at the element center.
     * @param iterations number of times erosion is applied.
     * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE:  dilate, morphologyEx, getStructuringElement
     */
    public static void erode(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType) {
        erode_1(src.nativeObj, dst.nativeObj, kernel.nativeObj, anchor.x, anchor.y, iterations, borderType);
    }

    /**
     * Erodes an image by using a specific structuring element.
     *
     * The function erodes the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the minimum is taken:
     *
     * \(\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for erosion; if {@code element=Mat()}, a {@code 3 x 3} rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement.
     * @param anchor position of the anchor within the element; default value (-1, -1) means that the
     * anchor is at the element center.
     * @param iterations number of times erosion is applied.
     * SEE:  dilate, morphologyEx, getStructuringElement
     */
    public static void erode(Mat src, Mat dst, Mat kernel, Point anchor, int iterations) {
        erode_2(src.nativeObj, dst.nativeObj, kernel.nativeObj, anchor.x, anchor.y, iterations);
    }

    /**
     * Erodes an image by using a specific structuring element.
     *
     * The function erodes the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the minimum is taken:
     *
     * \(\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for erosion; if {@code element=Mat()}, a {@code 3 x 3} rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement.
     * @param anchor position of the anchor within the element; default value (-1, -1) means that the
     * anchor is at the element center.
     * SEE:  dilate, morphologyEx, getStructuringElement
     */
    public static void erode(Mat src, Mat dst, Mat kernel, Point anchor) {
        erode_3(src.nativeObj, dst.nativeObj, kernel.nativeObj, anchor.x, anchor.y);
    }

    /**
     * Erodes an image by using a specific structuring element.
     *
     * The function erodes the source image using the specified structuring element that determines the
     * shape of a pixel neighborhood over which the minimum is taken:
     *
     * \(\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\)
     *
     * The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In
     * case of multi-channel images, each channel is processed independently.
     *
     * @param src input image; the number of channels can be arbitrary, but the depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst output image of the same size and type as src.
     * @param kernel structuring element used for erosion; if {@code element=Mat()}, a {@code 3 x 3} rectangular
     * structuring element is used. Kernel can be created using #getStructuringElement.
     * anchor is at the element center.
     * SEE:  dilate, morphologyEx, getStructuringElement
     */
    public static void erode(Mat src, Mat dst, Mat kernel) {
        erode_4(src.nativeObj, dst.nativeObj, kernel.nativeObj);
    }


    //
    // C++:  void cv::fillConvexPoly(Mat& img, vector_Point points, Scalar color, int lineType = LINE_8, int shift = 0)
    //

    /**
     * Fills a convex polygon.
     *
     * The function cv::fillConvexPoly draws a filled convex polygon. This function is much faster than the
     * function #fillPoly . It can fill not only convex polygons but any monotonic polygon without
     * self-intersections, that is, a polygon whose contour intersects every horizontal line (scan line)
     * twice at the most (though, its top-most and/or the bottom edge could be horizontal).
     *
     * @param img Image.
     * @param points Polygon vertices.
     * @param color Polygon color.
     * @param lineType Type of the polygon boundaries. See #LineTypes
     * @param shift Number of fractional bits in the vertex coordinates.
     */
    public static void fillConvexPoly(Mat img, MatOfPoint points, Scalar color, int lineType, int shift) {
        Mat points_mat = points;
        fillConvexPoly_0(img.nativeObj, points_mat.nativeObj, color.val[0], color.val[1], color.val[2], color.val[3], lineType, shift);
    }

    /**
     * Fills a convex polygon.
     *
     * The function cv::fillConvexPoly draws a filled convex polygon. This function is much faster than the
     * function #fillPoly . It can fill not only convex polygons but any monotonic polygon without
     * self-intersections, that is, a polygon whose contour intersects every horizontal line (scan line)
     * twice at the most (though, its top-most and/or the bottom edge could be horizontal).
     *
     * @param img Image.
     * @param points Polygon vertices.
     * @param color Polygon color.
     * @param lineType Type of the polygon boundaries. See #LineTypes
     */
    public static void fillConvexPoly(Mat img, MatOfPoint points, Scalar color, int lineType) {
        Mat points_mat = points;
        fillConvexPoly_1(img.nativeObj, points_mat.nativeObj, color.val[0], color.val[1], color.val[2], color.val[3], lineType);
    }

    /**
     * Fills a convex polygon.
     *
     * The function cv::fillConvexPoly draws a filled convex polygon. This function is much faster than the
     * function #fillPoly . It can fill not only convex polygons but any monotonic polygon without
     * self-intersections, that is, a polygon whose contour intersects every horizontal line (scan line)
     * twice at the most (though, its top-most and/or the bottom edge could be horizontal).
     *
     * @param img Image.
     * @param points Polygon vertices.
     * @param color Polygon color.
     */
    public static void fillConvexPoly(Mat img, MatOfPoint points, Scalar color) {
        Mat points_mat = points;
        fillConvexPoly_2(img.nativeObj, points_mat.nativeObj, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::fillPoly(Mat& img, vector_vector_Point pts, Scalar color, int lineType = LINE_8, int shift = 0, Point offset = Point())
    //

    /**
     * Fills the area bounded by one or more polygons.
     *
     * The function cv::fillPoly fills an area bounded by several polygonal contours. The function can fill
     * complex areas, for example, areas with holes, contours with self-intersections (some of their
     * parts), and so forth.
     *
     * @param img Image.
     * @param pts Array of polygons where each polygon is represented as an array of points.
     * @param color Polygon color.
     * @param lineType Type of the polygon boundaries. See #LineTypes
     * @param shift Number of fractional bits in the vertex coordinates.
     * @param offset Optional offset of all points of the contours.
     */
    public static void fillPoly(Mat img, List<MatOfPoint> pts, Scalar color, int lineType, int shift, Point offset) {
        List<Mat> pts_tmplm = new ArrayList<Mat>((pts != null) ? pts.size() : 0);
        Mat pts_mat = Converters.vector_vector_Point_to_Mat(pts, pts_tmplm);
        fillPoly_0(img.nativeObj, pts_mat.nativeObj, color.val[0], color.val[1], color.val[2], color.val[3], lineType, shift, offset.x, offset.y);
    }

    /**
     * Fills the area bounded by one or more polygons.
     *
     * The function cv::fillPoly fills an area bounded by several polygonal contours. The function can fill
     * complex areas, for example, areas with holes, contours with self-intersections (some of their
     * parts), and so forth.
     *
     * @param img Image.
     * @param pts Array of polygons where each polygon is represented as an array of points.
     * @param color Polygon color.
     * @param lineType Type of the polygon boundaries. See #LineTypes
     * @param shift Number of fractional bits in the vertex coordinates.
     */
    public static void fillPoly(Mat img, List<MatOfPoint> pts, Scalar color, int lineType, int shift) {
        List<Mat> pts_tmplm = new ArrayList<Mat>((pts != null) ? pts.size() : 0);
        Mat pts_mat = Converters.vector_vector_Point_to_Mat(pts, pts_tmplm);
        fillPoly_1(img.nativeObj, pts_mat.nativeObj, color.val[0], color.val[1], color.val[2], color.val[3], lineType, shift);
    }

    /**
     * Fills the area bounded by one or more polygons.
     *
     * The function cv::fillPoly fills an area bounded by several polygonal contours. The function can fill
     * complex areas, for example, areas with holes, contours with self-intersections (some of their
     * parts), and so forth.
     *
     * @param img Image.
     * @param pts Array of polygons where each polygon is represented as an array of points.
     * @param color Polygon color.
     * @param lineType Type of the polygon boundaries. See #LineTypes
     */
    public static void fillPoly(Mat img, List<MatOfPoint> pts, Scalar color, int lineType) {
        List<Mat> pts_tmplm = new ArrayList<Mat>((pts != null) ? pts.size() : 0);
        Mat pts_mat = Converters.vector_vector_Point_to_Mat(pts, pts_tmplm);
        fillPoly_2(img.nativeObj, pts_mat.nativeObj, color.val[0], color.val[1], color.val[2], color.val[3], lineType);
    }

    /**
     * Fills the area bounded by one or more polygons.
     *
     * The function cv::fillPoly fills an area bounded by several polygonal contours. The function can fill
     * complex areas, for example, areas with holes, contours with self-intersections (some of their
     * parts), and so forth.
     *
     * @param img Image.
     * @param pts Array of polygons where each polygon is represented as an array of points.
     * @param color Polygon color.
     */
    public static void fillPoly(Mat img, List<MatOfPoint> pts, Scalar color) {
        List<Mat> pts_tmplm = new ArrayList<Mat>((pts != null) ? pts.size() : 0);
        Mat pts_mat = Converters.vector_vector_Point_to_Mat(pts, pts_tmplm);
        fillPoly_3(img.nativeObj, pts_mat.nativeObj, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::filter2D(Mat src, Mat& dst, int ddepth, Mat kernel, Point anchor = Point(-1,-1), double delta = 0, int borderType = BORDER_DEFAULT)
    //

    /**
     * Convolves an image with the kernel.
     *
     * The function applies an arbitrary linear filter to an image. In-place operation is supported. When
     * the aperture is partially outside the image, the function interpolates outlier pixel values
     * according to the specified border mode.
     *
     * The function does actually compute correlation, not the convolution:
     *
     * \(\texttt{dst} (x,y) =  \sum _{ \stackrel{0\leq x' &lt; \texttt{kernel.cols},}{0\leq y' &lt; \texttt{kernel.rows}} }  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )\)
     *
     * That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
     * the kernel using #flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
     * anchor.y - 1)`.
     *
     * The function uses the DFT-based algorithm in case of sufficiently large kernels (~{@code 11 x 11} or
     * larger) and the direct algorithm for small kernels.
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src.
     * @param ddepth desired depth of the destination image, see REF: filter_depths "combinations"
     * @param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point
     * matrix; if you want to apply different kernels to different channels, split the image into
     * separate color planes using split and process them individually.
     * @param anchor anchor of the kernel that indicates the relative position of a filtered point within
     * the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor
     * is at the kernel center.
     * @param delta optional value added to the filtered pixels before storing them in dst.
     * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE:  sepFilter2D, dft, matchTemplate
     */
    public static void filter2D(Mat src, Mat dst, int ddepth, Mat kernel, Point anchor, double delta, int borderType) {
        filter2D_0(src.nativeObj, dst.nativeObj, ddepth, kernel.nativeObj, anchor.x, anchor.y, delta, borderType);
    }

    /**
     * Convolves an image with the kernel.
     *
     * The function applies an arbitrary linear filter to an image. In-place operation is supported. When
     * the aperture is partially outside the image, the function interpolates outlier pixel values
     * according to the specified border mode.
     *
     * The function does actually compute correlation, not the convolution:
     *
     * \(\texttt{dst} (x,y) =  \sum _{ \stackrel{0\leq x' &lt; \texttt{kernel.cols},}{0\leq y' &lt; \texttt{kernel.rows}} }  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )\)
     *
     * That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
     * the kernel using #flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
     * anchor.y - 1)`.
     *
     * The function uses the DFT-based algorithm in case of sufficiently large kernels (~{@code 11 x 11} or
     * larger) and the direct algorithm for small kernels.
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src.
     * @param ddepth desired depth of the destination image, see REF: filter_depths "combinations"
     * @param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point
     * matrix; if you want to apply different kernels to different channels, split the image into
     * separate color planes using split and process them individually.
     * @param anchor anchor of the kernel that indicates the relative position of a filtered point within
     * the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor
     * is at the kernel center.
     * @param delta optional value added to the filtered pixels before storing them in dst.
     * SEE:  sepFilter2D, dft, matchTemplate
     */
    public static void filter2D(Mat src, Mat dst, int ddepth, Mat kernel, Point anchor, double delta) {
        filter2D_1(src.nativeObj, dst.nativeObj, ddepth, kernel.nativeObj, anchor.x, anchor.y, delta);
    }

    /**
     * Convolves an image with the kernel.
     *
     * The function applies an arbitrary linear filter to an image. In-place operation is supported. When
     * the aperture is partially outside the image, the function interpolates outlier pixel values
     * according to the specified border mode.
     *
     * The function does actually compute correlation, not the convolution:
     *
     * \(\texttt{dst} (x,y) =  \sum _{ \stackrel{0\leq x' &lt; \texttt{kernel.cols},}{0\leq y' &lt; \texttt{kernel.rows}} }  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )\)
     *
     * That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
     * the kernel using #flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
     * anchor.y - 1)`.
     *
     * The function uses the DFT-based algorithm in case of sufficiently large kernels (~{@code 11 x 11} or
     * larger) and the direct algorithm for small kernels.
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src.
     * @param ddepth desired depth of the destination image, see REF: filter_depths "combinations"
     * @param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point
     * matrix; if you want to apply different kernels to different channels, split the image into
     * separate color planes using split and process them individually.
     * @param anchor anchor of the kernel that indicates the relative position of a filtered point within
     * the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor
     * is at the kernel center.
     * SEE:  sepFilter2D, dft, matchTemplate
     */
    public static void filter2D(Mat src, Mat dst, int ddepth, Mat kernel, Point anchor) {
        filter2D_2(src.nativeObj, dst.nativeObj, ddepth, kernel.nativeObj, anchor.x, anchor.y);
    }

    /**
     * Convolves an image with the kernel.
     *
     * The function applies an arbitrary linear filter to an image. In-place operation is supported. When
     * the aperture is partially outside the image, the function interpolates outlier pixel values
     * according to the specified border mode.
     *
     * The function does actually compute correlation, not the convolution:
     *
     * \(\texttt{dst} (x,y) =  \sum _{ \stackrel{0\leq x' &lt; \texttt{kernel.cols},}{0\leq y' &lt; \texttt{kernel.rows}} }  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )\)
     *
     * That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
     * the kernel using #flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
     * anchor.y - 1)`.
     *
     * The function uses the DFT-based algorithm in case of sufficiently large kernels (~{@code 11 x 11} or
     * larger) and the direct algorithm for small kernels.
     *
     * @param src input image.
     * @param dst output image of the same size and the same number of channels as src.
     * @param ddepth desired depth of the destination image, see REF: filter_depths "combinations"
     * @param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point
     * matrix; if you want to apply different kernels to different channels, split the image into
     * separate color planes using split and process them individually.
     * the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor
     * is at the kernel center.
     * SEE:  sepFilter2D, dft, matchTemplate
     */
    public static void filter2D(Mat src, Mat dst, int ddepth, Mat kernel) {
        filter2D_3(src.nativeObj, dst.nativeObj, ddepth, kernel.nativeObj);
    }


    //
    // C++:  void cv::findContours(Mat image, vector_vector_Point& contours, Mat& hierarchy, int mode, int method, Point offset = Point())
    //

    /**
     * Finds contours in a binary image.
     *
     * The function retrieves contours from the binary image using the algorithm CITE: Suzuki85 . The contours
     * are a useful tool for shape analysis and object detection and recognition. See squares.cpp in the
     * OpenCV sample directory.
     * <b>Note:</b> Since opencv 3.2 source image is not modified by this function.
     *
     * @param image Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero
     * pixels remain 0's, so the image is treated as binary . You can use #compare, #inRange, #threshold ,
     * #adaptiveThreshold, #Canny, and others to create a binary image out of a grayscale or color one.
     * If mode equals to #RETR_CCOMP or #RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1).
     * @param contours Detected contours. Each contour is stored as a vector of points (e.g.
     * std::vector&lt;std::vector&lt;cv::Point&gt; &gt;).
     * @param hierarchy Optional output vector (e.g. std::vector&lt;cv::Vec4i&gt;), containing information about the image topology. It has
     * as many elements as the number of contours. For each i-th contour contours[i], the elements
     * hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3] are set to 0-based indices
     * in contours of the next and previous contours at the same hierarchical level, the first child
     * contour and the parent contour, respectively. If for the contour i there are no next, previous,
     * parent, or nested contours, the corresponding elements of hierarchy[i] will be negative.
     * @param mode Contour retrieval mode, see #RetrievalModes
     * @param method Contour approximation method, see #ContourApproximationModes
     * @param offset Optional offset by which every contour point is shifted. This is useful if the
     * contours are extracted from the image ROI and then they should be analyzed in the whole image
     * context.
     */
    public static void findContours(Mat image, List<MatOfPoint> contours, Mat hierarchy, int mode, int method, Point offset) {
        Mat contours_mat = new Mat();
        findContours_0(image.nativeObj, contours_mat.nativeObj, hierarchy.nativeObj, mode, method, offset.x, offset.y);
        Converters.Mat_to_vector_vector_Point(contours_mat, contours);
        contours_mat.release();
    }

    /**
     * Finds contours in a binary image.
     *
     * The function retrieves contours from the binary image using the algorithm CITE: Suzuki85 . The contours
     * are a useful tool for shape analysis and object detection and recognition. See squares.cpp in the
     * OpenCV sample directory.
     * <b>Note:</b> Since opencv 3.2 source image is not modified by this function.
     *
     * @param image Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero
     * pixels remain 0's, so the image is treated as binary . You can use #compare, #inRange, #threshold ,
     * #adaptiveThreshold, #Canny, and others to create a binary image out of a grayscale or color one.
     * If mode equals to #RETR_CCOMP or #RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1).
     * @param contours Detected contours. Each contour is stored as a vector of points (e.g.
     * std::vector&lt;std::vector&lt;cv::Point&gt; &gt;).
     * @param hierarchy Optional output vector (e.g. std::vector&lt;cv::Vec4i&gt;), containing information about the image topology. It has
     * as many elements as the number of contours. For each i-th contour contours[i], the elements
     * hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3] are set to 0-based indices
     * in contours of the next and previous contours at the same hierarchical level, the first child
     * contour and the parent contour, respectively. If for the contour i there are no next, previous,
     * parent, or nested contours, the corresponding elements of hierarchy[i] will be negative.
     * @param mode Contour retrieval mode, see #RetrievalModes
     * @param method Contour approximation method, see #ContourApproximationModes
     * contours are extracted from the image ROI and then they should be analyzed in the whole image
     * context.
     */
    public static void findContours(Mat image, List<MatOfPoint> contours, Mat hierarchy, int mode, int method) {
        Mat contours_mat = new Mat();
        findContours_1(image.nativeObj, contours_mat.nativeObj, hierarchy.nativeObj, mode, method);
        Converters.Mat_to_vector_vector_Point(contours_mat, contours);
        contours_mat.release();
    }


    //
    // C++:  void cv::fitLine(Mat points, Mat& line, int distType, double param, double reps, double aeps)
    //

    /**
     * Fits a line to a 2D or 3D point set.
     *
     * The function fitLine fits a line to a 2D or 3D point set by minimizing \(\sum_i \rho(r_i)\) where
     * \(r_i\) is a distance between the \(i^{th}\) point, the line and \(\rho(r)\) is a distance function, one
     * of the following:
     * <ul>
     *   <li>
     *   DIST_L2
     * \(\rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)}\)
     *   </li>
     *   <li>
     *  DIST_L1
     * \(\rho (r) = r\)
     *   </li>
     *   <li>
     *  DIST_L12
     * \(\rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)\)
     *   </li>
     *   <li>
     *  DIST_FAIR
     * \(\rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998\)
     *   </li>
     *   <li>
     *  DIST_WELSCH
     * \(\rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846\)
     *   </li>
     *   <li>
     *  DIST_HUBER
     * \(\rho (r) =  \fork{r^2/2}{if \(r &lt; C\)}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345\)
     *   </li>
     * </ul>
     *
     * The algorithm is based on the M-estimator ( &lt;http://en.wikipedia.org/wiki/M-estimator&gt; ) technique
     * that iteratively fits the line using the weighted least-squares algorithm. After each iteration the
     * weights \(w_i\) are adjusted to be inversely proportional to \(\rho(r_i)\) .
     *
     * @param points Input vector of 2D or 3D points, stored in std::vector&lt;&gt; or Mat.
     * @param line Output line parameters. In case of 2D fitting, it should be a vector of 4 elements
     * (like Vec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and
     * (x0, y0) is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like
     * Vec6f) - (vx, vy, vz, x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line
     * and (x0, y0, z0) is a point on the line.
     * @param distType Distance used by the M-estimator, see #DistanceTypes
     * @param param Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value
     * is chosen.
     * @param reps Sufficient accuracy for the radius (distance between the coordinate origin and the line).
     * @param aeps Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.
     */
    public static void fitLine(Mat points, Mat line, int distType, double param, double reps, double aeps) {
        fitLine_0(points.nativeObj, line.nativeObj, distType, param, reps, aeps);
    }


    //
    // C++:  void cv::getDerivKernels(Mat& kx, Mat& ky, int dx, int dy, int ksize, bool normalize = false, int ktype = CV_32F)
    //

    /**
     * Returns filter coefficients for computing spatial image derivatives.
     *
     * The function computes and returns the filter coefficients for spatial image derivatives. When
     * {@code ksize=FILTER_SCHARR}, the Scharr \(3 \times 3\) kernels are generated (see #Scharr). Otherwise, Sobel
     * kernels are generated (see #Sobel). The filters are normally passed to #sepFilter2D or to
     *
     * @param kx Output matrix of row filter coefficients. It has the type ktype .
     * @param ky Output matrix of column filter coefficients. It has the type ktype .
     * @param dx Derivative order in respect of x.
     * @param dy Derivative order in respect of y.
     * @param ksize Aperture size. It can be FILTER_SCHARR, 1, 3, 5, or 7.
     * @param normalize Flag indicating whether to normalize (scale down) the filter coefficients or not.
     * Theoretically, the coefficients should have the denominator \(=2^{ksize*2-dx-dy-2}\). If you are
     * going to filter floating-point images, you are likely to use the normalized kernels. But if you
     * compute derivatives of an 8-bit image, store the results in a 16-bit image, and wish to preserve
     * all the fractional bits, you may want to set normalize=false .
     * @param ktype Type of filter coefficients. It can be CV_32f or CV_64F .
     */
    public static void getDerivKernels(Mat kx, Mat ky, int dx, int dy, int ksize, boolean normalize, int ktype) {
        getDerivKernels_0(kx.nativeObj, ky.nativeObj, dx, dy, ksize, normalize, ktype);
    }

    /**
     * Returns filter coefficients for computing spatial image derivatives.
     *
     * The function computes and returns the filter coefficients for spatial image derivatives. When
     * {@code ksize=FILTER_SCHARR}, the Scharr \(3 \times 3\) kernels are generated (see #Scharr). Otherwise, Sobel
     * kernels are generated (see #Sobel). The filters are normally passed to #sepFilter2D or to
     *
     * @param kx Output matrix of row filter coefficients. It has the type ktype .
     * @param ky Output matrix of column filter coefficients. It has the type ktype .
     * @param dx Derivative order in respect of x.
     * @param dy Derivative order in respect of y.
     * @param ksize Aperture size. It can be FILTER_SCHARR, 1, 3, 5, or 7.
     * @param normalize Flag indicating whether to normalize (scale down) the filter coefficients or not.
     * Theoretically, the coefficients should have the denominator \(=2^{ksize*2-dx-dy-2}\). If you are
     * going to filter floating-point images, you are likely to use the normalized kernels. But if you
     * compute derivatives of an 8-bit image, store the results in a 16-bit image, and wish to preserve
     * all the fractional bits, you may want to set normalize=false .
     */
    public static void getDerivKernels(Mat kx, Mat ky, int dx, int dy, int ksize, boolean normalize) {
        getDerivKernels_1(kx.nativeObj, ky.nativeObj, dx, dy, ksize, normalize);
    }

    /**
     * Returns filter coefficients for computing spatial image derivatives.
     *
     * The function computes and returns the filter coefficients for spatial image derivatives. When
     * {@code ksize=FILTER_SCHARR}, the Scharr \(3 \times 3\) kernels are generated (see #Scharr). Otherwise, Sobel
     * kernels are generated (see #Sobel). The filters are normally passed to #sepFilter2D or to
     *
     * @param kx Output matrix of row filter coefficients. It has the type ktype .
     * @param ky Output matrix of column filter coefficients. It has the type ktype .
     * @param dx Derivative order in respect of x.
     * @param dy Derivative order in respect of y.
     * @param ksize Aperture size. It can be FILTER_SCHARR, 1, 3, 5, or 7.
     * Theoretically, the coefficients should have the denominator \(=2^{ksize*2-dx-dy-2}\). If you are
     * going to filter floating-point images, you are likely to use the normalized kernels. But if you
     * compute derivatives of an 8-bit image, store the results in a 16-bit image, and wish to preserve
     * all the fractional bits, you may want to set normalize=false .
     */
    public static void getDerivKernels(Mat kx, Mat ky, int dx, int dy, int ksize) {
        getDerivKernels_2(kx.nativeObj, ky.nativeObj, dx, dy, ksize);
    }


    //
    // C++:  void cv::getRectSubPix(Mat image, Size patchSize, Point2f center, Mat& patch, int patchType = -1)
    //

    /**
     * Retrieves a pixel rectangle from an image with sub-pixel accuracy.
     *
     * The function getRectSubPix extracts pixels from src:
     *
     * \(patch(x, y) = src(x +  \texttt{center.x} - ( \texttt{dst.cols} -1)*0.5, y +  \texttt{center.y} - ( \texttt{dst.rows} -1)*0.5)\)
     *
     * where the values of the pixels at non-integer coordinates are retrieved using bilinear
     * interpolation. Every channel of multi-channel images is processed independently. Also
     * the image should be a single channel or three channel image. While the center of the
     * rectangle must be inside the image, parts of the rectangle may be outside.
     *
     * @param image Source image.
     * @param patchSize Size of the extracted patch.
     * @param center Floating point coordinates of the center of the extracted rectangle within the
     * source image. The center must be inside the image.
     * @param patch Extracted patch that has the size patchSize and the same number of channels as src .
     * @param patchType Depth of the extracted pixels. By default, they have the same depth as src .
     *
     * SEE:  warpAffine, warpPerspective
     */
    public static void getRectSubPix(Mat image, Size patchSize, Point center, Mat patch, int patchType) {
        getRectSubPix_0(image.nativeObj, patchSize.width, patchSize.height, center.x, center.y, patch.nativeObj, patchType);
    }

    /**
     * Retrieves a pixel rectangle from an image with sub-pixel accuracy.
     *
     * The function getRectSubPix extracts pixels from src:
     *
     * \(patch(x, y) = src(x +  \texttt{center.x} - ( \texttt{dst.cols} -1)*0.5, y +  \texttt{center.y} - ( \texttt{dst.rows} -1)*0.5)\)
     *
     * where the values of the pixels at non-integer coordinates are retrieved using bilinear
     * interpolation. Every channel of multi-channel images is processed independently. Also
     * the image should be a single channel or three channel image. While the center of the
     * rectangle must be inside the image, parts of the rectangle may be outside.
     *
     * @param image Source image.
     * @param patchSize Size of the extracted patch.
     * @param center Floating point coordinates of the center of the extracted rectangle within the
     * source image. The center must be inside the image.
     * @param patch Extracted patch that has the size patchSize and the same number of channels as src .
     *
     * SEE:  warpAffine, warpPerspective
     */
    public static void getRectSubPix(Mat image, Size patchSize, Point center, Mat patch) {
        getRectSubPix_1(image.nativeObj, patchSize.width, patchSize.height, center.x, center.y, patch.nativeObj);
    }


    //
    // C++:  void cv::goodFeaturesToTrack(Mat image, vector_Point& corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize, int gradientSize, bool useHarrisDetector = false, double k = 0.04)
    //

    public static void goodFeaturesToTrack(Mat image, MatOfPoint corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize, int gradientSize, boolean useHarrisDetector, double k) {
        Mat corners_mat = corners;
        goodFeaturesToTrack_0(image.nativeObj, corners_mat.nativeObj, maxCorners, qualityLevel, minDistance, mask.nativeObj, blockSize, gradientSize, useHarrisDetector, k);
    }

    public static void goodFeaturesToTrack(Mat image, MatOfPoint corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize, int gradientSize, boolean useHarrisDetector) {
        Mat corners_mat = corners;
        goodFeaturesToTrack_1(image.nativeObj, corners_mat.nativeObj, maxCorners, qualityLevel, minDistance, mask.nativeObj, blockSize, gradientSize, useHarrisDetector);
    }

    public static void goodFeaturesToTrack(Mat image, MatOfPoint corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize, int gradientSize) {
        Mat corners_mat = corners;
        goodFeaturesToTrack_2(image.nativeObj, corners_mat.nativeObj, maxCorners, qualityLevel, minDistance, mask.nativeObj, blockSize, gradientSize);
    }


    //
    // C++:  void cv::goodFeaturesToTrack(Mat image, vector_Point& corners, int maxCorners, double qualityLevel, double minDistance, Mat mask = Mat(), int blockSize = 3, bool useHarrisDetector = false, double k = 0.04)
    //

    /**
     * Determines strong corners on an image.
     *
     * The function finds the most prominent corners in the image or in the specified image region, as
     * described in CITE: Shi94
     *
     * <ul>
     *   <li>
     *    Function calculates the corner quality measure at every source image pixel using the
     *     #cornerMinEigenVal or #cornerHarris .
     *   </li>
     *   <li>
     *    Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
     *     retained).
     *   </li>
     *   <li>
     *    The corners with the minimal eigenvalue less than
     *     \(\texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y)\) are rejected.
     *   </li>
     *   <li>
     *    The remaining corners are sorted by the quality measure in the descending order.
     *   </li>
     *   <li>
     *    Function throws away each corner for which there is a stronger corner at a distance less than
     *     maxDistance.
     *   </li>
     * </ul>
     *
     * The function can be used to initialize a point-based tracker of an object.
     *
     * <b>Note:</b> If the function is called with different values A and B of the parameter qualityLevel , and
     * A &gt; B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector
     * with qualityLevel=B .
     *
     * @param image Input 8-bit or floating-point 32-bit, single-channel image.
     * @param corners Output vector of detected corners.
     * @param maxCorners Maximum number of corners to return. If there are more corners than are found,
     * the strongest of them is returned. {@code maxCorners &lt;= 0} implies that no limit on the maximum is set
     * and all detected corners are returned.
     * @param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
     * parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
     * (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the
     * quality measure less than the product are rejected. For example, if the best corner has the
     * quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
     * less than 15 are rejected.
     * @param minDistance Minimum possible Euclidean distance between the returned corners.
     * @param mask Optional region of interest. If the image is not empty (it needs to have the type
     * CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
     * @param blockSize Size of an average block for computing a derivative covariation matrix over each
     * pixel neighborhood. See cornerEigenValsAndVecs .
     * @param useHarrisDetector Parameter indicating whether to use a Harris detector (see #cornerHarris)
     * or #cornerMinEigenVal.
     * @param k Free parameter of the Harris detector.
     *
     * SEE:  cornerMinEigenVal, cornerHarris, calcOpticalFlowPyrLK, estimateRigidTransform,
     */
    public static void goodFeaturesToTrack(Mat image, MatOfPoint corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize, boolean useHarrisDetector, double k) {
        Mat corners_mat = corners;
        goodFeaturesToTrack_3(image.nativeObj, corners_mat.nativeObj, maxCorners, qualityLevel, minDistance, mask.nativeObj, blockSize, useHarrisDetector, k);
    }

    /**
     * Determines strong corners on an image.
     *
     * The function finds the most prominent corners in the image or in the specified image region, as
     * described in CITE: Shi94
     *
     * <ul>
     *   <li>
     *    Function calculates the corner quality measure at every source image pixel using the
     *     #cornerMinEigenVal or #cornerHarris .
     *   </li>
     *   <li>
     *    Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
     *     retained).
     *   </li>
     *   <li>
     *    The corners with the minimal eigenvalue less than
     *     \(\texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y)\) are rejected.
     *   </li>
     *   <li>
     *    The remaining corners are sorted by the quality measure in the descending order.
     *   </li>
     *   <li>
     *    Function throws away each corner for which there is a stronger corner at a distance less than
     *     maxDistance.
     *   </li>
     * </ul>
     *
     * The function can be used to initialize a point-based tracker of an object.
     *
     * <b>Note:</b> If the function is called with different values A and B of the parameter qualityLevel , and
     * A &gt; B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector
     * with qualityLevel=B .
     *
     * @param image Input 8-bit or floating-point 32-bit, single-channel image.
     * @param corners Output vector of detected corners.
     * @param maxCorners Maximum number of corners to return. If there are more corners than are found,
     * the strongest of them is returned. {@code maxCorners &lt;= 0} implies that no limit on the maximum is set
     * and all detected corners are returned.
     * @param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
     * parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
     * (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the
     * quality measure less than the product are rejected. For example, if the best corner has the
     * quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
     * less than 15 are rejected.
     * @param minDistance Minimum possible Euclidean distance between the returned corners.
     * @param mask Optional region of interest. If the image is not empty (it needs to have the type
     * CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
     * @param blockSize Size of an average block for computing a derivative covariation matrix over each
     * pixel neighborhood. See cornerEigenValsAndVecs .
     * @param useHarrisDetector Parameter indicating whether to use a Harris detector (see #cornerHarris)
     * or #cornerMinEigenVal.
     *
     * SEE:  cornerMinEigenVal, cornerHarris, calcOpticalFlowPyrLK, estimateRigidTransform,
     */
    public static void goodFeaturesToTrack(Mat image, MatOfPoint corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize, boolean useHarrisDetector) {
        Mat corners_mat = corners;
        goodFeaturesToTrack_4(image.nativeObj, corners_mat.nativeObj, maxCorners, qualityLevel, minDistance, mask.nativeObj, blockSize, useHarrisDetector);
    }

    /**
     * Determines strong corners on an image.
     *
     * The function finds the most prominent corners in the image or in the specified image region, as
     * described in CITE: Shi94
     *
     * <ul>
     *   <li>
     *    Function calculates the corner quality measure at every source image pixel using the
     *     #cornerMinEigenVal or #cornerHarris .
     *   </li>
     *   <li>
     *    Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
     *     retained).
     *   </li>
     *   <li>
     *    The corners with the minimal eigenvalue less than
     *     \(\texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y)\) are rejected.
     *   </li>
     *   <li>
     *    The remaining corners are sorted by the quality measure in the descending order.
     *   </li>
     *   <li>
     *    Function throws away each corner for which there is a stronger corner at a distance less than
     *     maxDistance.
     *   </li>
     * </ul>
     *
     * The function can be used to initialize a point-based tracker of an object.
     *
     * <b>Note:</b> If the function is called with different values A and B of the parameter qualityLevel , and
     * A &gt; B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector
     * with qualityLevel=B .
     *
     * @param image Input 8-bit or floating-point 32-bit, single-channel image.
     * @param corners Output vector of detected corners.
     * @param maxCorners Maximum number of corners to return. If there are more corners than are found,
     * the strongest of them is returned. {@code maxCorners &lt;= 0} implies that no limit on the maximum is set
     * and all detected corners are returned.
     * @param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
     * parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
     * (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the
     * quality measure less than the product are rejected. For example, if the best corner has the
     * quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
     * less than 15 are rejected.
     * @param minDistance Minimum possible Euclidean distance between the returned corners.
     * @param mask Optional region of interest. If the image is not empty (it needs to have the type
     * CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
     * @param blockSize Size of an average block for computing a derivative covariation matrix over each
     * pixel neighborhood. See cornerEigenValsAndVecs .
     * or #cornerMinEigenVal.
     *
     * SEE:  cornerMinEigenVal, cornerHarris, calcOpticalFlowPyrLK, estimateRigidTransform,
     */
    public static void goodFeaturesToTrack(Mat image, MatOfPoint corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize) {
        Mat corners_mat = corners;
        goodFeaturesToTrack_5(image.nativeObj, corners_mat.nativeObj, maxCorners, qualityLevel, minDistance, mask.nativeObj, blockSize);
    }

    /**
     * Determines strong corners on an image.
     *
     * The function finds the most prominent corners in the image or in the specified image region, as
     * described in CITE: Shi94
     *
     * <ul>
     *   <li>
     *    Function calculates the corner quality measure at every source image pixel using the
     *     #cornerMinEigenVal or #cornerHarris .
     *   </li>
     *   <li>
     *    Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
     *     retained).
     *   </li>
     *   <li>
     *    The corners with the minimal eigenvalue less than
     *     \(\texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y)\) are rejected.
     *   </li>
     *   <li>
     *    The remaining corners are sorted by the quality measure in the descending order.
     *   </li>
     *   <li>
     *    Function throws away each corner for which there is a stronger corner at a distance less than
     *     maxDistance.
     *   </li>
     * </ul>
     *
     * The function can be used to initialize a point-based tracker of an object.
     *
     * <b>Note:</b> If the function is called with different values A and B of the parameter qualityLevel , and
     * A &gt; B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector
     * with qualityLevel=B .
     *
     * @param image Input 8-bit or floating-point 32-bit, single-channel image.
     * @param corners Output vector of detected corners.
     * @param maxCorners Maximum number of corners to return. If there are more corners than are found,
     * the strongest of them is returned. {@code maxCorners &lt;= 0} implies that no limit on the maximum is set
     * and all detected corners are returned.
     * @param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
     * parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
     * (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the
     * quality measure less than the product are rejected. For example, if the best corner has the
     * quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
     * less than 15 are rejected.
     * @param minDistance Minimum possible Euclidean distance between the returned corners.
     * @param mask Optional region of interest. If the image is not empty (it needs to have the type
     * CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
     * pixel neighborhood. See cornerEigenValsAndVecs .
     * or #cornerMinEigenVal.
     *
     * SEE:  cornerMinEigenVal, cornerHarris, calcOpticalFlowPyrLK, estimateRigidTransform,
     */
    public static void goodFeaturesToTrack(Mat image, MatOfPoint corners, int maxCorners, double qualityLevel, double minDistance, Mat mask) {
        Mat corners_mat = corners;
        goodFeaturesToTrack_6(image.nativeObj, corners_mat.nativeObj, maxCorners, qualityLevel, minDistance, mask.nativeObj);
    }

    /**
     * Determines strong corners on an image.
     *
     * The function finds the most prominent corners in the image or in the specified image region, as
     * described in CITE: Shi94
     *
     * <ul>
     *   <li>
     *    Function calculates the corner quality measure at every source image pixel using the
     *     #cornerMinEigenVal or #cornerHarris .
     *   </li>
     *   <li>
     *    Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
     *     retained).
     *   </li>
     *   <li>
     *    The corners with the minimal eigenvalue less than
     *     \(\texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y)\) are rejected.
     *   </li>
     *   <li>
     *    The remaining corners are sorted by the quality measure in the descending order.
     *   </li>
     *   <li>
     *    Function throws away each corner for which there is a stronger corner at a distance less than
     *     maxDistance.
     *   </li>
     * </ul>
     *
     * The function can be used to initialize a point-based tracker of an object.
     *
     * <b>Note:</b> If the function is called with different values A and B of the parameter qualityLevel , and
     * A &gt; B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector
     * with qualityLevel=B .
     *
     * @param image Input 8-bit or floating-point 32-bit, single-channel image.
     * @param corners Output vector of detected corners.
     * @param maxCorners Maximum number of corners to return. If there are more corners than are found,
     * the strongest of them is returned. {@code maxCorners &lt;= 0} implies that no limit on the maximum is set
     * and all detected corners are returned.
     * @param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
     * parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
     * (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the
     * quality measure less than the product are rejected. For example, if the best corner has the
     * quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
     * less than 15 are rejected.
     * @param minDistance Minimum possible Euclidean distance between the returned corners.
     * CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
     * pixel neighborhood. See cornerEigenValsAndVecs .
     * or #cornerMinEigenVal.
     *
     * SEE:  cornerMinEigenVal, cornerHarris, calcOpticalFlowPyrLK, estimateRigidTransform,
     */
    public static void goodFeaturesToTrack(Mat image, MatOfPoint corners, int maxCorners, double qualityLevel, double minDistance) {
        Mat corners_mat = corners;
        goodFeaturesToTrack_7(image.nativeObj, corners_mat.nativeObj, maxCorners, qualityLevel, minDistance);
    }


    //
    // C++:  void cv::grabCut(Mat img, Mat& mask, Rect rect, Mat& bgdModel, Mat& fgdModel, int iterCount, int mode = GC_EVAL)
    //

    /**
     * Runs the GrabCut algorithm.
     *
     * The function implements the [GrabCut image segmentation algorithm](http://en.wikipedia.org/wiki/GrabCut).
     *
     * @param img Input 8-bit 3-channel image.
     * @param mask Input/output 8-bit single-channel mask. The mask is initialized by the function when
     * mode is set to #GC_INIT_WITH_RECT. Its elements may have one of the #GrabCutClasses.
     * @param rect ROI containing a segmented object. The pixels outside of the ROI are marked as
     * "obvious background". The parameter is only used when mode==#GC_INIT_WITH_RECT .
     * @param bgdModel Temporary array for the background model. Do not modify it while you are
     * processing the same image.
     * @param fgdModel Temporary arrays for the foreground model. Do not modify it while you are
     * processing the same image.
     * @param iterCount Number of iterations the algorithm should make before returning the result. Note
     * that the result can be refined with further calls with mode==#GC_INIT_WITH_MASK or
     * mode==GC_EVAL .
     * @param mode Operation mode that could be one of the #GrabCutModes
     */
    public static void grabCut(Mat img, Mat mask, Rect rect, Mat bgdModel, Mat fgdModel, int iterCount, int mode) {
        grabCut_0(img.nativeObj, mask.nativeObj, rect.x, rect.y, rect.width, rect.height, bgdModel.nativeObj, fgdModel.nativeObj, iterCount, mode);
    }

    /**
     * Runs the GrabCut algorithm.
     *
     * The function implements the [GrabCut image segmentation algorithm](http://en.wikipedia.org/wiki/GrabCut).
     *
     * @param img Input 8-bit 3-channel image.
     * @param mask Input/output 8-bit single-channel mask. The mask is initialized by the function when
     * mode is set to #GC_INIT_WITH_RECT. Its elements may have one of the #GrabCutClasses.
     * @param rect ROI containing a segmented object. The pixels outside of the ROI are marked as
     * "obvious background". The parameter is only used when mode==#GC_INIT_WITH_RECT .
     * @param bgdModel Temporary array for the background model. Do not modify it while you are
     * processing the same image.
     * @param fgdModel Temporary arrays for the foreground model. Do not modify it while you are
     * processing the same image.
     * @param iterCount Number of iterations the algorithm should make before returning the result. Note
     * that the result can be refined with further calls with mode==#GC_INIT_WITH_MASK or
     * mode==GC_EVAL .
     */
    public static void grabCut(Mat img, Mat mask, Rect rect, Mat bgdModel, Mat fgdModel, int iterCount) {
        grabCut_1(img.nativeObj, mask.nativeObj, rect.x, rect.y, rect.width, rect.height, bgdModel.nativeObj, fgdModel.nativeObj, iterCount);
    }


    //
    // C++:  void cv::integral(Mat src, Mat& sum, Mat& sqsum, Mat& tilted, int sdepth = -1, int sqdepth = -1)
    //

    /**
     * Calculates the integral of an image.
     *
     * The function calculates one or more integral images for the source image as follows:
     *
     * \(\texttt{sum} (X,Y) =  \sum _{x&lt;X,y&lt;Y}  \texttt{image} (x,y)\)
     *
     * \(\texttt{sqsum} (X,Y) =  \sum _{x&lt;X,y&lt;Y}  \texttt{image} (x,y)^2\)
     *
     * \(\texttt{tilted} (X,Y) =  \sum _{y&lt;Y,abs(x-X+1) \leq Y-y-1}  \texttt{image} (x,y)\)
     *
     * Using these integral images, you can calculate sum, mean, and standard deviation over a specific
     * up-right or rotated rectangular region of the image in a constant time, for example:
     *
     * \(\sum _{x_1 \leq x &lt; x_2,  \, y_1  \leq y &lt; y_2}  \texttt{image} (x,y) =  \texttt{sum} (x_2,y_2)- \texttt{sum} (x_1,y_2)- \texttt{sum} (x_2,y_1)+ \texttt{sum} (x_1,y_1)\)
     *
     * It makes possible to do a fast blurring or fast block correlation with a variable window size, for
     * example. In case of multi-channel images, sums for each channel are accumulated independently.
     *
     * As a practical example, the next figure shows the calculation of the integral of a straight
     * rectangle Rect(3,3,3,2) and of a tilted rectangle Rect(5,1,2,3) . The selected pixels in the
     * original image are shown, as well as the relative pixels in the integral images sum and tilted .
     *
     * ![integral calculation example](pics/integral.png)
     *
     * @param src input image as \(W \times H\), 8-bit or floating-point (32f or 64f).
     * @param sum integral image as \((W+1)\times (H+1)\) , 32-bit integer or floating-point (32f or 64f).
     * @param sqsum integral image for squared pixel values; it is \((W+1)\times (H+1)\), double-precision
     * floating-point (64f) array.
     * @param tilted integral for the image rotated by 45 degrees; it is \((W+1)\times (H+1)\) array with
     * the same data type as sum.
     * @param sdepth desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or
     * CV_64F.
     * @param sqdepth desired depth of the integral image of squared pixel values, CV_32F or CV_64F.
     */
    public static void integral3(Mat src, Mat sum, Mat sqsum, Mat tilted, int sdepth, int sqdepth) {
        integral3_0(src.nativeObj, sum.nativeObj, sqsum.nativeObj, tilted.nativeObj, sdepth, sqdepth);
    }

    /**
     * Calculates the integral of an image.
     *
     * The function calculates one or more integral images for the source image as follows:
     *
     * \(\texttt{sum} (X,Y) =  \sum _{x&lt;X,y&lt;Y}  \texttt{image} (x,y)\)
     *
     * \(\texttt{sqsum} (X,Y) =  \sum _{x&lt;X,y&lt;Y}  \texttt{image} (x,y)^2\)
     *
     * \(\texttt{tilted} (X,Y) =  \sum _{y&lt;Y,abs(x-X+1) \leq Y-y-1}  \texttt{image} (x,y)\)
     *
     * Using these integral images, you can calculate sum, mean, and standard deviation over a specific
     * up-right or rotated rectangular region of the image in a constant time, for example:
     *
     * \(\sum _{x_1 \leq x &lt; x_2,  \, y_1  \leq y &lt; y_2}  \texttt{image} (x,y) =  \texttt{sum} (x_2,y_2)- \texttt{sum} (x_1,y_2)- \texttt{sum} (x_2,y_1)+ \texttt{sum} (x_1,y_1)\)
     *
     * It makes possible to do a fast blurring or fast block correlation with a variable window size, for
     * example. In case of multi-channel images, sums for each channel are accumulated independently.
     *
     * As a practical example, the next figure shows the calculation of the integral of a straight
     * rectangle Rect(3,3,3,2) and of a tilted rectangle Rect(5,1,2,3) . The selected pixels in the
     * original image are shown, as well as the relative pixels in the integral images sum and tilted .
     *
     * ![integral calculation example](pics/integral.png)
     *
     * @param src input image as \(W \times H\), 8-bit or floating-point (32f or 64f).
     * @param sum integral image as \((W+1)\times (H+1)\) , 32-bit integer or floating-point (32f or 64f).
     * @param sqsum integral image for squared pixel values; it is \((W+1)\times (H+1)\), double-precision
     * floating-point (64f) array.
     * @param tilted integral for the image rotated by 45 degrees; it is \((W+1)\times (H+1)\) array with
     * the same data type as sum.
     * @param sdepth desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or
     * CV_64F.
     */
    public static void integral3(Mat src, Mat sum, Mat sqsum, Mat tilted, int sdepth) {
        integral3_1(src.nativeObj, sum.nativeObj, sqsum.nativeObj, tilted.nativeObj, sdepth);
    }

    /**
     * Calculates the integral of an image.
     *
     * The function calculates one or more integral images for the source image as follows:
     *
     * \(\texttt{sum} (X,Y) =  \sum _{x&lt;X,y&lt;Y}  \texttt{image} (x,y)\)
     *
     * \(\texttt{sqsum} (X,Y) =  \sum _{x&lt;X,y&lt;Y}  \texttt{image} (x,y)^2\)
     *
     * \(\texttt{tilted} (X,Y) =  \sum _{y&lt;Y,abs(x-X+1) \leq Y-y-1}  \texttt{image} (x,y)\)
     *
     * Using these integral images, you can calculate sum, mean, and standard deviation over a specific
     * up-right or rotated rectangular region of the image in a constant time, for example:
     *
     * \(\sum _{x_1 \leq x &lt; x_2,  \, y_1  \leq y &lt; y_2}  \texttt{image} (x,y) =  \texttt{sum} (x_2,y_2)- \texttt{sum} (x_1,y_2)- \texttt{sum} (x_2,y_1)+ \texttt{sum} (x_1,y_1)\)
     *
     * It makes possible to do a fast blurring or fast block correlation with a variable window size, for
     * example. In case of multi-channel images, sums for each channel are accumulated independently.
     *
     * As a practical example, the next figure shows the calculation of the integral of a straight
     * rectangle Rect(3,3,3,2) and of a tilted rectangle Rect(5,1,2,3) . The selected pixels in the
     * original image are shown, as well as the relative pixels in the integral images sum and tilted .
     *
     * ![integral calculation example](pics/integral.png)
     *
     * @param src input image as \(W \times H\), 8-bit or floating-point (32f or 64f).
     * @param sum integral image as \((W+1)\times (H+1)\) , 32-bit integer or floating-point (32f or 64f).
     * @param sqsum integral image for squared pixel values; it is \((W+1)\times (H+1)\), double-precision
     * floating-point (64f) array.
     * @param tilted integral for the image rotated by 45 degrees; it is \((W+1)\times (H+1)\) array with
     * the same data type as sum.
     * CV_64F.
     */
    public static void integral3(Mat src, Mat sum, Mat sqsum, Mat tilted) {
        integral3_2(src.nativeObj, sum.nativeObj, sqsum.nativeObj, tilted.nativeObj);
    }


    //
    // C++:  void cv::integral(Mat src, Mat& sum, Mat& sqsum, int sdepth = -1, int sqdepth = -1)
    //

    public static void integral2(Mat src, Mat sum, Mat sqsum, int sdepth, int sqdepth) {
        integral2_0(src.nativeObj, sum.nativeObj, sqsum.nativeObj, sdepth, sqdepth);
    }

    public static void integral2(Mat src, Mat sum, Mat sqsum, int sdepth) {
        integral2_1(src.nativeObj, sum.nativeObj, sqsum.nativeObj, sdepth);
    }

    public static void integral2(Mat src, Mat sum, Mat sqsum) {
        integral2_2(src.nativeObj, sum.nativeObj, sqsum.nativeObj);
    }


    //
    // C++:  void cv::integral(Mat src, Mat& sum, int sdepth = -1)
    //

    public static void integral(Mat src, Mat sum, int sdepth) {
        integral_0(src.nativeObj, sum.nativeObj, sdepth);
    }

    public static void integral(Mat src, Mat sum) {
        integral_1(src.nativeObj, sum.nativeObj);
    }


    //
    // C++:  void cv::invertAffineTransform(Mat M, Mat& iM)
    //

    /**
     * Inverts an affine transformation.
     *
     * The function computes an inverse affine transformation represented by \(2 \times 3\) matrix M:
     *
     * \(\begin{bmatrix} a_{11} &amp; a_{12} &amp; b_1  \\ a_{21} &amp; a_{22} &amp; b_2 \end{bmatrix}\)
     *
     * The result is also a \(2 \times 3\) matrix of the same type as M.
     *
     * @param M Original affine transformation.
     * @param iM Output reverse affine transformation.
     */
    public static void invertAffineTransform(Mat M, Mat iM) {
        invertAffineTransform_0(M.nativeObj, iM.nativeObj);
    }


    //
    // C++:  void cv::line(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    //

    /**
     * Draws a line segment connecting two points.
     *
     * The function line draws the line segment between pt1 and pt2 points in the image. The line is
     * clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected
     * or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased
     * lines are drawn using Gaussian filtering.
     *
     * @param img Image.
     * @param pt1 First point of the line segment.
     * @param pt2 Second point of the line segment.
     * @param color Line color.
     * @param thickness Line thickness.
     * @param lineType Type of the line. See #LineTypes.
     * @param shift Number of fractional bits in the point coordinates.
     */
    public static void line(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int lineType, int shift) {
        line_0(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, shift);
    }

    /**
     * Draws a line segment connecting two points.
     *
     * The function line draws the line segment between pt1 and pt2 points in the image. The line is
     * clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected
     * or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased
     * lines are drawn using Gaussian filtering.
     *
     * @param img Image.
     * @param pt1 First point of the line segment.
     * @param pt2 Second point of the line segment.
     * @param color Line color.
     * @param thickness Line thickness.
     * @param lineType Type of the line. See #LineTypes.
     */
    public static void line(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int lineType) {
        line_1(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     * Draws a line segment connecting two points.
     *
     * The function line draws the line segment between pt1 and pt2 points in the image. The line is
     * clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected
     * or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased
     * lines are drawn using Gaussian filtering.
     *
     * @param img Image.
     * @param pt1 First point of the line segment.
     * @param pt2 Second point of the line segment.
     * @param color Line color.
     * @param thickness Line thickness.
     */
    public static void line(Mat img, Point pt1, Point pt2, Scalar color, int thickness) {
        line_2(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     * Draws a line segment connecting two points.
     *
     * The function line draws the line segment between pt1 and pt2 points in the image. The line is
     * clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected
     * or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased
     * lines are drawn using Gaussian filtering.
     *
     * @param img Image.
     * @param pt1 First point of the line segment.
     * @param pt2 Second point of the line segment.
     * @param color Line color.
     */
    public static void line(Mat img, Point pt1, Point pt2, Scalar color) {
        line_3(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::linearPolar(Mat src, Mat& dst, Point2f center, double maxRadius, int flags)
    //

    /**
     * Remaps an image to polar coordinates space.
     *
     * @deprecated This function produces same result as cv::warpPolar(src, dst, src.size(), center, maxRadius, flags)
     *
     *
     * Transform the source image using the following transformation (See REF: polar_remaps_reference_image "Polar remaps reference image c)"):
     * \(\begin{array}{l}
     *   dst( \rho , \phi ) = src(x,y) \\
     *   dst.size() \leftarrow src.size()
     * \end{array}\)
     *
     * where
     * \(\begin{array}{l}
     *   I = (dx,dy) = (x - center.x,y - center.y) \\
     *   \rho = Kmag \cdot \texttt{magnitude} (I) ,\\
     *   \phi = angle \cdot \texttt{angle} (I)
     * \end{array}\)
     *
     * and
     * \(\begin{array}{l}
     *   Kx = src.cols / maxRadius \\
     *   Ky = src.rows / 2\Pi
     * \end{array}\)
     *
     *
     * @param src Source image
     * @param dst Destination image. It will have same size and type as src.
     * @param center The transformation center;
     * @param maxRadius The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.
     * @param flags A combination of interpolation methods, see #InterpolationFlags
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *    The function can not operate in-place.
     *   </li>
     *   <li>
     *    To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.
     *   </li>
     * </ul>
     *
     * SEE: cv::logPolar
     */
    @Deprecated
    public static void linearPolar(Mat src, Mat dst, Point center, double maxRadius, int flags) {
        linearPolar_0(src.nativeObj, dst.nativeObj, center.x, center.y, maxRadius, flags);
    }


    //
    // C++:  void cv::logPolar(Mat src, Mat& dst, Point2f center, double M, int flags)
    //

    /**
     * Remaps an image to semilog-polar coordinates space.
     *
     * @deprecated This function produces same result as cv::warpPolar(src, dst, src.size(), center, maxRadius, flags+WARP_POLAR_LOG);
     *
     *
     * Transform the source image using the following transformation (See REF: polar_remaps_reference_image "Polar remaps reference image d)"):
     * \(\begin{array}{l}
     *   dst( \rho , \phi ) = src(x,y) \\
     *   dst.size() \leftarrow src.size()
     * \end{array}\)
     *
     * where
     * \(\begin{array}{l}
     *   I = (dx,dy) = (x - center.x,y - center.y) \\
     *   \rho = M \cdot log_e(\texttt{magnitude} (I)) ,\\
     *   \phi = Kangle \cdot \texttt{angle} (I) \\
     * \end{array}\)
     *
     * and
     * \(\begin{array}{l}
     *   M = src.cols / log_e(maxRadius) \\
     *   Kangle = src.rows / 2\Pi \\
     * \end{array}\)
     *
     * The function emulates the human "foveal" vision and can be used for fast scale and
     * rotation-invariant template matching, for object tracking and so forth.
     * @param src Source image
     * @param dst Destination image. It will have same size and type as src.
     * @param center The transformation center; where the output precision is maximal
     * @param M Magnitude scale parameter. It determines the radius of the bounding circle to transform too.
     * @param flags A combination of interpolation methods, see #InterpolationFlags
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *    The function can not operate in-place.
     *   </li>
     *   <li>
     *    To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.
     *   </li>
     * </ul>
     *
     * SEE: cv::linearPolar
     */
    @Deprecated
    public static void logPolar(Mat src, Mat dst, Point center, double M, int flags) {
        logPolar_0(src.nativeObj, dst.nativeObj, center.x, center.y, M, flags);
    }


    //
    // C++:  void cv::matchTemplate(Mat image, Mat templ, Mat& result, int method, Mat mask = Mat())
    //

    /**
     * Compares a template against overlapped image regions.
     *
     * The function slides through image , compares the overlapped patches of size \(w \times h\) against
     * templ using the specified method and stores the comparison results in result . Here are the formulae
     * for the available comparison methods ( \(I\) denotes image, \(T\) template, \(R\) result ). The summation
     * is done over template and/or the image patch: \(x' = 0...w-1, y' = 0...h-1\)
     *
     * After the function finishes the comparison, the best matches can be found as global minimums (when
     * #TM_SQDIFF was used) or maximums (when #TM_CCORR or #TM_CCOEFF was used) using the
     * #minMaxLoc function. In case of a color image, template summation in the numerator and each sum in
     * the denominator is done over all of the channels and separate mean values are used for each channel.
     * That is, the function can take a color template and a color image. The result will still be a
     * single-channel image, which is easier to analyze.
     *
     * @param image Image where the search is running. It must be 8-bit or 32-bit floating-point.
     * @param templ Searched template. It must be not greater than the source image and have the same
     * data type.
     * @param result Map of comparison results. It must be single-channel 32-bit floating-point. If image
     * is \(W \times H\) and templ is \(w \times h\) , then result is \((W-w+1) \times (H-h+1)\) .
     * @param method Parameter specifying the comparison method, see #TemplateMatchModes
     * @param mask Mask of searched template. It must have the same datatype and size with templ. It is
     * not set by default. Currently, only the #TM_SQDIFF and #TM_CCORR_NORMED methods are supported.
     */
    public static void matchTemplate(Mat image, Mat templ, Mat result, int method, Mat mask) {
        matchTemplate_0(image.nativeObj, templ.nativeObj, result.nativeObj, method, mask.nativeObj);
    }

    /**
     * Compares a template against overlapped image regions.
     *
     * The function slides through image , compares the overlapped patches of size \(w \times h\) against
     * templ using the specified method and stores the comparison results in result . Here are the formulae
     * for the available comparison methods ( \(I\) denotes image, \(T\) template, \(R\) result ). The summation
     * is done over template and/or the image patch: \(x' = 0...w-1, y' = 0...h-1\)
     *
     * After the function finishes the comparison, the best matches can be found as global minimums (when
     * #TM_SQDIFF was used) or maximums (when #TM_CCORR or #TM_CCOEFF was used) using the
     * #minMaxLoc function. In case of a color image, template summation in the numerator and each sum in
     * the denominator is done over all of the channels and separate mean values are used for each channel.
     * That is, the function can take a color template and a color image. The result will still be a
     * single-channel image, which is easier to analyze.
     *
     * @param image Image where the search is running. It must be 8-bit or 32-bit floating-point.
     * @param templ Searched template. It must be not greater than the source image and have the same
     * data type.
     * @param result Map of comparison results. It must be single-channel 32-bit floating-point. If image
     * is \(W \times H\) and templ is \(w \times h\) , then result is \((W-w+1) \times (H-h+1)\) .
     * @param method Parameter specifying the comparison method, see #TemplateMatchModes
     * not set by default. Currently, only the #TM_SQDIFF and #TM_CCORR_NORMED methods are supported.
     */
    public static void matchTemplate(Mat image, Mat templ, Mat result, int method) {
        matchTemplate_1(image.nativeObj, templ.nativeObj, result.nativeObj, method);
    }


    //
    // C++:  void cv::medianBlur(Mat src, Mat& dst, int ksize)
    //

    /**
     * Blurs an image using the median filter.
     *
     * The function smoothes an image using the median filter with the \(\texttt{ksize} \times
     * \texttt{ksize}\) aperture. Each channel of a multi-channel image is processed independently.
     * In-place operation is supported.
     *
     * <b>Note:</b> The median filter uses #BORDER_REPLICATE internally to cope with border pixels, see #BorderTypes
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be
     * CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param dst destination array of the same size and type as src.
     * @param ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
     * SEE:  bilateralFilter, blur, boxFilter, GaussianBlur
     */
    public static void medianBlur(Mat src, Mat dst, int ksize) {
        medianBlur_0(src.nativeObj, dst.nativeObj, ksize);
    }


    //
    // C++:  void cv::minEnclosingCircle(vector_Point2f points, Point2f& center, float& radius)
    //

    /**
     * Finds a circle of the minimum area enclosing a 2D point set.
     *
     * The function finds the minimal enclosing circle of a 2D point set using an iterative algorithm.
     *
     * @param points Input vector of 2D points, stored in std::vector&lt;&gt; or Mat
     * @param center Output center of the circle.
     * @param radius Output radius of the circle.
     */
    public static void minEnclosingCircle(MatOfPoint2f points, Point center, float[] radius) {
        Mat points_mat = points;
        double[] center_out = new double[2];
        double[] radius_out = new double[1];
        minEnclosingCircle_0(points_mat.nativeObj, center_out, radius_out);
        if(center!=null){ center.x = center_out[0]; center.y = center_out[1]; } 
        if(radius!=null) radius[0] = (float)radius_out[0];
    }


    //
    // C++:  void cv::morphologyEx(Mat src, Mat& dst, int op, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
    //

    /**
     * Performs advanced morphological transformations.
     *
     * The function cv::morphologyEx can perform advanced morphological transformations using an erosion and dilation as
     * basic operations.
     *
     * Any of the operations can be done in-place. In case of multi-channel images, each channel is
     * processed independently.
     *
     * @param src Source image. The number of channels can be arbitrary. The depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst Destination image of the same size and type as source image.
     * @param op Type of a morphological operation, see #MorphTypes
     * @param kernel Structuring element. It can be created using #getStructuringElement.
     * @param anchor Anchor position with the kernel. Negative values mean that the anchor is at the
     * kernel center.
     * @param iterations Number of times erosion and dilation are applied.
     * @param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * @param borderValue Border value in case of a constant border. The default value has a special
     * meaning.
     * SEE:  dilate, erode, getStructuringElement
     * <b>Note:</b> The number of iterations is the number of times erosion or dilatation operation will be applied.
     * For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply
     * successively: erode -&gt; erode -&gt; dilate -&gt; dilate (and not erode -&gt; dilate -&gt; erode -&gt; dilate).
     */
    public static void morphologyEx(Mat src, Mat dst, int op, Mat kernel, Point anchor, int iterations, int borderType, Scalar borderValue) {
        morphologyEx_0(src.nativeObj, dst.nativeObj, op, kernel.nativeObj, anchor.x, anchor.y, iterations, borderType, borderValue.val[0], borderValue.val[1], borderValue.val[2], borderValue.val[3]);
    }

    /**
     * Performs advanced morphological transformations.
     *
     * The function cv::morphologyEx can perform advanced morphological transformations using an erosion and dilation as
     * basic operations.
     *
     * Any of the operations can be done in-place. In case of multi-channel images, each channel is
     * processed independently.
     *
     * @param src Source image. The number of channels can be arbitrary. The depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst Destination image of the same size and type as source image.
     * @param op Type of a morphological operation, see #MorphTypes
     * @param kernel Structuring element. It can be created using #getStructuringElement.
     * @param anchor Anchor position with the kernel. Negative values mean that the anchor is at the
     * kernel center.
     * @param iterations Number of times erosion and dilation are applied.
     * @param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * meaning.
     * SEE:  dilate, erode, getStructuringElement
     * <b>Note:</b> The number of iterations is the number of times erosion or dilatation operation will be applied.
     * For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply
     * successively: erode -&gt; erode -&gt; dilate -&gt; dilate (and not erode -&gt; dilate -&gt; erode -&gt; dilate).
     */
    public static void morphologyEx(Mat src, Mat dst, int op, Mat kernel, Point anchor, int iterations, int borderType) {
        morphologyEx_1(src.nativeObj, dst.nativeObj, op, kernel.nativeObj, anchor.x, anchor.y, iterations, borderType);
    }

    /**
     * Performs advanced morphological transformations.
     *
     * The function cv::morphologyEx can perform advanced morphological transformations using an erosion and dilation as
     * basic operations.
     *
     * Any of the operations can be done in-place. In case of multi-channel images, each channel is
     * processed independently.
     *
     * @param src Source image. The number of channels can be arbitrary. The depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst Destination image of the same size and type as source image.
     * @param op Type of a morphological operation, see #MorphTypes
     * @param kernel Structuring element. It can be created using #getStructuringElement.
     * @param anchor Anchor position with the kernel. Negative values mean that the anchor is at the
     * kernel center.
     * @param iterations Number of times erosion and dilation are applied.
     * meaning.
     * SEE:  dilate, erode, getStructuringElement
     * <b>Note:</b> The number of iterations is the number of times erosion or dilatation operation will be applied.
     * For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply
     * successively: erode -&gt; erode -&gt; dilate -&gt; dilate (and not erode -&gt; dilate -&gt; erode -&gt; dilate).
     */
    public static void morphologyEx(Mat src, Mat dst, int op, Mat kernel, Point anchor, int iterations) {
        morphologyEx_2(src.nativeObj, dst.nativeObj, op, kernel.nativeObj, anchor.x, anchor.y, iterations);
    }

    /**
     * Performs advanced morphological transformations.
     *
     * The function cv::morphologyEx can perform advanced morphological transformations using an erosion and dilation as
     * basic operations.
     *
     * Any of the operations can be done in-place. In case of multi-channel images, each channel is
     * processed independently.
     *
     * @param src Source image. The number of channels can be arbitrary. The depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst Destination image of the same size and type as source image.
     * @param op Type of a morphological operation, see #MorphTypes
     * @param kernel Structuring element. It can be created using #getStructuringElement.
     * @param anchor Anchor position with the kernel. Negative values mean that the anchor is at the
     * kernel center.
     * meaning.
     * SEE:  dilate, erode, getStructuringElement
     * <b>Note:</b> The number of iterations is the number of times erosion or dilatation operation will be applied.
     * For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply
     * successively: erode -&gt; erode -&gt; dilate -&gt; dilate (and not erode -&gt; dilate -&gt; erode -&gt; dilate).
     */
    public static void morphologyEx(Mat src, Mat dst, int op, Mat kernel, Point anchor) {
        morphologyEx_3(src.nativeObj, dst.nativeObj, op, kernel.nativeObj, anchor.x, anchor.y);
    }

    /**
     * Performs advanced morphological transformations.
     *
     * The function cv::morphologyEx can perform advanced morphological transformations using an erosion and dilation as
     * basic operations.
     *
     * Any of the operations can be done in-place. In case of multi-channel images, each channel is
     * processed independently.
     *
     * @param src Source image. The number of channels can be arbitrary. The depth should be one of
     * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     * @param dst Destination image of the same size and type as source image.
     * @param op Type of a morphological operation, see #MorphTypes
     * @param kernel Structuring element. It can be created using #getStructuringElement.
     * kernel center.
     * meaning.
     * SEE:  dilate, erode, getStructuringElement
     * <b>Note:</b> The number of iterations is the number of times erosion or dilatation operation will be applied.
     * For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply
     * successively: erode -&gt; erode -&gt; dilate -&gt; dilate (and not erode -&gt; dilate -&gt; erode -&gt; dilate).
     */
    public static void morphologyEx(Mat src, Mat dst, int op, Mat kernel) {
        morphologyEx_4(src.nativeObj, dst.nativeObj, op, kernel.nativeObj);
    }


    //
    // C++:  void cv::polylines(Mat& img, vector_vector_Point pts, bool isClosed, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    //

    /**
     * Draws several polygonal curves.
     *
     * @param img Image.
     * @param pts Array of polygonal curves.
     * @param isClosed Flag indicating whether the drawn polylines are closed or not. If they are closed,
     * the function draws a line from the last vertex of each curve to its first vertex.
     * @param color Polyline color.
     * @param thickness Thickness of the polyline edges.
     * @param lineType Type of the line segments. See #LineTypes
     * @param shift Number of fractional bits in the vertex coordinates.
     *
     * The function cv::polylines draws one or more polygonal curves.
     */
    public static void polylines(Mat img, List<MatOfPoint> pts, boolean isClosed, Scalar color, int thickness, int lineType, int shift) {
        List<Mat> pts_tmplm = new ArrayList<Mat>((pts != null) ? pts.size() : 0);
        Mat pts_mat = Converters.vector_vector_Point_to_Mat(pts, pts_tmplm);
        polylines_0(img.nativeObj, pts_mat.nativeObj, isClosed, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, shift);
    }

    /**
     * Draws several polygonal curves.
     *
     * @param img Image.
     * @param pts Array of polygonal curves.
     * @param isClosed Flag indicating whether the drawn polylines are closed or not. If they are closed,
     * the function draws a line from the last vertex of each curve to its first vertex.
     * @param color Polyline color.
     * @param thickness Thickness of the polyline edges.
     * @param lineType Type of the line segments. See #LineTypes
     *
     * The function cv::polylines draws one or more polygonal curves.
     */
    public static void polylines(Mat img, List<MatOfPoint> pts, boolean isClosed, Scalar color, int thickness, int lineType) {
        List<Mat> pts_tmplm = new ArrayList<Mat>((pts != null) ? pts.size() : 0);
        Mat pts_mat = Converters.vector_vector_Point_to_Mat(pts, pts_tmplm);
        polylines_1(img.nativeObj, pts_mat.nativeObj, isClosed, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     * Draws several polygonal curves.
     *
     * @param img Image.
     * @param pts Array of polygonal curves.
     * @param isClosed Flag indicating whether the drawn polylines are closed or not. If they are closed,
     * the function draws a line from the last vertex of each curve to its first vertex.
     * @param color Polyline color.
     * @param thickness Thickness of the polyline edges.
     *
     * The function cv::polylines draws one or more polygonal curves.
     */
    public static void polylines(Mat img, List<MatOfPoint> pts, boolean isClosed, Scalar color, int thickness) {
        List<Mat> pts_tmplm = new ArrayList<Mat>((pts != null) ? pts.size() : 0);
        Mat pts_mat = Converters.vector_vector_Point_to_Mat(pts, pts_tmplm);
        polylines_2(img.nativeObj, pts_mat.nativeObj, isClosed, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     * Draws several polygonal curves.
     *
     * @param img Image.
     * @param pts Array of polygonal curves.
     * @param isClosed Flag indicating whether the drawn polylines are closed or not. If they are closed,
     * the function draws a line from the last vertex of each curve to its first vertex.
     * @param color Polyline color.
     *
     * The function cv::polylines draws one or more polygonal curves.
     */
    public static void polylines(Mat img, List<MatOfPoint> pts, boolean isClosed, Scalar color) {
        List<Mat> pts_tmplm = new ArrayList<Mat>((pts != null) ? pts.size() : 0);
        Mat pts_mat = Converters.vector_vector_Point_to_Mat(pts, pts_tmplm);
        polylines_3(img.nativeObj, pts_mat.nativeObj, isClosed, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::preCornerDetect(Mat src, Mat& dst, int ksize, int borderType = BORDER_DEFAULT)
    //

    /**
     * Calculates a feature map for corner detection.
     *
     * The function calculates the complex spatial derivative-based function of the source image
     *
     * \(\texttt{dst} = (D_x  \texttt{src} )^2  \cdot D_{yy}  \texttt{src} + (D_y  \texttt{src} )^2  \cdot D_{xx}  \texttt{src} - 2 D_x  \texttt{src} \cdot D_y  \texttt{src} \cdot D_{xy}  \texttt{src}\)
     *
     * where \(D_x\),\(D_y\) are the first image derivatives, \(D_{xx}\),\(D_{yy}\) are the second image
     * derivatives, and \(D_{xy}\) is the mixed derivative.
     *
     * The corners can be found as local maximums of the functions, as shown below:
     * <code>
     *     Mat corners, dilated_corners;
     *     preCornerDetect(image, corners, 3);
     *     // dilation with 3x3 rectangular structuring element
     *     dilate(corners, dilated_corners, Mat(), 1);
     *     Mat corner_mask = corners == dilated_corners;
     * </code>
     *
     * @param src Source single-channel 8-bit of floating-point image.
     * @param dst Output image that has the type CV_32F and the same size as src .
     * @param ksize %Aperture size of the Sobel .
     * @param borderType Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.
     */
    public static void preCornerDetect(Mat src, Mat dst, int ksize, int borderType) {
        preCornerDetect_0(src.nativeObj, dst.nativeObj, ksize, borderType);
    }

    /**
     * Calculates a feature map for corner detection.
     *
     * The function calculates the complex spatial derivative-based function of the source image
     *
     * \(\texttt{dst} = (D_x  \texttt{src} )^2  \cdot D_{yy}  \texttt{src} + (D_y  \texttt{src} )^2  \cdot D_{xx}  \texttt{src} - 2 D_x  \texttt{src} \cdot D_y  \texttt{src} \cdot D_{xy}  \texttt{src}\)
     *
     * where \(D_x\),\(D_y\) are the first image derivatives, \(D_{xx}\),\(D_{yy}\) are the second image
     * derivatives, and \(D_{xy}\) is the mixed derivative.
     *
     * The corners can be found as local maximums of the functions, as shown below:
     * <code>
     *     Mat corners, dilated_corners;
     *     preCornerDetect(image, corners, 3);
     *     // dilation with 3x3 rectangular structuring element
     *     dilate(corners, dilated_corners, Mat(), 1);
     *     Mat corner_mask = corners == dilated_corners;
     * </code>
     *
     * @param src Source single-channel 8-bit of floating-point image.
     * @param dst Output image that has the type CV_32F and the same size as src .
     * @param ksize %Aperture size of the Sobel .
     */
    public static void preCornerDetect(Mat src, Mat dst, int ksize) {
        preCornerDetect_1(src.nativeObj, dst.nativeObj, ksize);
    }


    //
    // C++:  void cv::putText(Mat& img, String text, Point org, int fontFace, double fontScale, Scalar color, int thickness = 1, int lineType = LINE_8, bool bottomLeftOrigin = false)
    //

    /**
     * Draws a text string.
     *
     * The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered
     * using the specified font are replaced by question marks. See #getTextSize for a text rendering code
     * example.
     *
     * @param img Image.
     * @param text Text string to be drawn.
     * @param org Bottom-left corner of the text string in the image.
     * @param fontFace Font type, see #HersheyFonts.
     * @param fontScale Font scale factor that is multiplied by the font-specific base size.
     * @param color Text color.
     * @param thickness Thickness of the lines used to draw a text.
     * @param lineType Line type. See #LineTypes
     * @param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise,
     * it is at the top-left corner.
     */
    public static void putText(Mat img, String text, Point org, int fontFace, double fontScale, Scalar color, int thickness, int lineType, boolean bottomLeftOrigin) {
        putText_0(img.nativeObj, text, org.x, org.y, fontFace, fontScale, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, bottomLeftOrigin);
    }

    /**
     * Draws a text string.
     *
     * The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered
     * using the specified font are replaced by question marks. See #getTextSize for a text rendering code
     * example.
     *
     * @param img Image.
     * @param text Text string to be drawn.
     * @param org Bottom-left corner of the text string in the image.
     * @param fontFace Font type, see #HersheyFonts.
     * @param fontScale Font scale factor that is multiplied by the font-specific base size.
     * @param color Text color.
     * @param thickness Thickness of the lines used to draw a text.
     * @param lineType Line type. See #LineTypes
     * it is at the top-left corner.
     */
    public static void putText(Mat img, String text, Point org, int fontFace, double fontScale, Scalar color, int thickness, int lineType) {
        putText_1(img.nativeObj, text, org.x, org.y, fontFace, fontScale, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     * Draws a text string.
     *
     * The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered
     * using the specified font are replaced by question marks. See #getTextSize for a text rendering code
     * example.
     *
     * @param img Image.
     * @param text Text string to be drawn.
     * @param org Bottom-left corner of the text string in the image.
     * @param fontFace Font type, see #HersheyFonts.
     * @param fontScale Font scale factor that is multiplied by the font-specific base size.
     * @param color Text color.
     * @param thickness Thickness of the lines used to draw a text.
     * it is at the top-left corner.
     */
    public static void putText(Mat img, String text, Point org, int fontFace, double fontScale, Scalar color, int thickness) {
        putText_2(img.nativeObj, text, org.x, org.y, fontFace, fontScale, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     * Draws a text string.
     *
     * The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered
     * using the specified font are replaced by question marks. See #getTextSize for a text rendering code
     * example.
     *
     * @param img Image.
     * @param text Text string to be drawn.
     * @param org Bottom-left corner of the text string in the image.
     * @param fontFace Font type, see #HersheyFonts.
     * @param fontScale Font scale factor that is multiplied by the font-specific base size.
     * @param color Text color.
     * it is at the top-left corner.
     */
    public static void putText(Mat img, String text, Point org, int fontFace, double fontScale, Scalar color) {
        putText_3(img.nativeObj, text, org.x, org.y, fontFace, fontScale, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::pyrDown(Mat src, Mat& dst, Size dstsize = Size(), int borderType = BORDER_DEFAULT)
    //

    /**
     * Blurs an image and downsamples it.
     *
     * By default, size of the output image is computed as {@code Size((src.cols+1)/2, (src.rows+1)/2)}, but in
     * any case, the following conditions should be satisfied:
     *
     * \(\begin{array}{l} | \texttt{dstsize.width} *2-src.cols| \leq 2 \\ | \texttt{dstsize.height} *2-src.rows| \leq 2 \end{array}\)
     *
     * The function performs the downsampling step of the Gaussian pyramid construction. First, it
     * convolves the source image with the kernel:
     *
     * \(\frac{1}{256} \begin{bmatrix} 1 &amp; 4 &amp; 6 &amp; 4 &amp; 1  \\ 4 &amp; 16 &amp; 24 &amp; 16 &amp; 4  \\ 6 &amp; 24 &amp; 36 &amp; 24 &amp; 6  \\ 4 &amp; 16 &amp; 24 &amp; 16 &amp; 4  \\ 1 &amp; 4 &amp; 6 &amp; 4 &amp; 1 \end{bmatrix}\)
     *
     * Then, it downsamples the image by rejecting even rows and columns.
     *
     * @param src input image.
     * @param dst output image; it has the specified size and the same type as src.
     * @param dstsize size of the output image.
     * @param borderType Pixel extrapolation method, see #BorderTypes (#BORDER_CONSTANT isn't supported)
     */
    public static void pyrDown(Mat src, Mat dst, Size dstsize, int borderType) {
        pyrDown_0(src.nativeObj, dst.nativeObj, dstsize.width, dstsize.height, borderType);
    }

    /**
     * Blurs an image and downsamples it.
     *
     * By default, size of the output image is computed as {@code Size((src.cols+1)/2, (src.rows+1)/2)}, but in
     * any case, the following conditions should be satisfied:
     *
     * \(\begin{array}{l} | \texttt{dstsize.width} *2-src.cols| \leq 2 \\ | \texttt{dstsize.height} *2-src.rows| \leq 2 \end{array}\)
     *
     * The function performs the downsampling step of the Gaussian pyramid construction. First, it
     * convolves the source image with the kernel:
     *
     * \(\frac{1}{256} \begin{bmatrix} 1 &amp; 4 &amp; 6 &amp; 4 &amp; 1  \\ 4 &amp; 16 &amp; 24 &amp; 16 &amp; 4  \\ 6 &amp; 24 &amp; 36 &amp; 24 &amp; 6  \\ 4 &amp; 16 &amp; 24 &amp; 16 &amp; 4  \\ 1 &amp; 4 &amp; 6 &amp; 4 &amp; 1 \end{bmatrix}\)
     *
     * Then, it downsamples the image by rejecting even rows and columns.
     *
     * @param src input image.
     * @param dst output image; it has the specified size and the same type as src.
     * @param dstsize size of the output image.
     */
    public static void pyrDown(Mat src, Mat dst, Size dstsize) {
        pyrDown_1(src.nativeObj, dst.nativeObj, dstsize.width, dstsize.height);
    }

    /**
     * Blurs an image and downsamples it.
     *
     * By default, size of the output image is computed as {@code Size((src.cols+1)/2, (src.rows+1)/2)}, but in
     * any case, the following conditions should be satisfied:
     *
     * \(\begin{array}{l} | \texttt{dstsize.width} *2-src.cols| \leq 2 \\ | \texttt{dstsize.height} *2-src.rows| \leq 2 \end{array}\)
     *
     * The function performs the downsampling step of the Gaussian pyramid construction. First, it
     * convolves the source image with the kernel:
     *
     * \(\frac{1}{256} \begin{bmatrix} 1 &amp; 4 &amp; 6 &amp; 4 &amp; 1  \\ 4 &amp; 16 &amp; 24 &amp; 16 &amp; 4  \\ 6 &amp; 24 &amp; 36 &amp; 24 &amp; 6  \\ 4 &amp; 16 &amp; 24 &amp; 16 &amp; 4  \\ 1 &amp; 4 &amp; 6 &amp; 4 &amp; 1 \end{bmatrix}\)
     *
     * Then, it downsamples the image by rejecting even rows and columns.
     *
     * @param src input image.
     * @param dst output image; it has the specified size and the same type as src.
     */
    public static void pyrDown(Mat src, Mat dst) {
        pyrDown_2(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::pyrMeanShiftFiltering(Mat src, Mat& dst, double sp, double sr, int maxLevel = 1, TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1))
    //

    /**
     * Performs initial step of meanshift segmentation of an image.
     *
     * The function implements the filtering stage of meanshift segmentation, that is, the output of the
     * function is the filtered "posterized" image with color gradients and fine-grain texture flattened.
     * At every pixel (X,Y) of the input image (or down-sized input image, see below) the function executes
     * meanshift iterations, that is, the pixel (X,Y) neighborhood in the joint space-color hyperspace is
     * considered:
     *
     * \((x,y): X- \texttt{sp} \le x  \le X+ \texttt{sp} , Y- \texttt{sp} \le y  \le Y+ \texttt{sp} , ||(R,G,B)-(r,g,b)||   \le \texttt{sr}\)
     *
     * where (R,G,B) and (r,g,b) are the vectors of color components at (X,Y) and (x,y), respectively
     * (though, the algorithm does not depend on the color space used, so any 3-component color space can
     * be used instead). Over the neighborhood the average spatial value (X',Y') and average color vector
     * (R',G',B') are found and they act as the neighborhood center on the next iteration:
     *
     * \((X,Y)~(X',Y'), (R,G,B)~(R',G',B').\)
     *
     * After the iterations over, the color components of the initial pixel (that is, the pixel from where
     * the iterations started) are set to the final value (average color at the last iteration):
     *
     * \(I(X,Y) &lt;- (R*,G*,B*)\)
     *
     * When maxLevel &gt; 0, the gaussian pyramid of maxLevel+1 levels is built, and the above procedure is
     * run on the smallest layer first. After that, the results are propagated to the larger layer and the
     * iterations are run again only on those pixels where the layer colors differ by more than sr from the
     * lower-resolution layer of the pyramid. That makes boundaries of color regions sharper. Note that the
     * results will be actually different from the ones obtained by running the meanshift procedure on the
     * whole original image (i.e. when maxLevel==0).
     *
     * @param src The source 8-bit, 3-channel image.
     * @param dst The destination image of the same format and the same size as the source.
     * @param sp The spatial window radius.
     * @param sr The color window radius.
     * @param maxLevel Maximum level of the pyramid for the segmentation.
     * @param termcrit Termination criteria: when to stop meanshift iterations.
     */
    public static void pyrMeanShiftFiltering(Mat src, Mat dst, double sp, double sr, int maxLevel, TermCriteria termcrit) {
        pyrMeanShiftFiltering_0(src.nativeObj, dst.nativeObj, sp, sr, maxLevel, termcrit.type, termcrit.maxCount, termcrit.epsilon);
    }

    /**
     * Performs initial step of meanshift segmentation of an image.
     *
     * The function implements the filtering stage of meanshift segmentation, that is, the output of the
     * function is the filtered "posterized" image with color gradients and fine-grain texture flattened.
     * At every pixel (X,Y) of the input image (or down-sized input image, see below) the function executes
     * meanshift iterations, that is, the pixel (X,Y) neighborhood in the joint space-color hyperspace is
     * considered:
     *
     * \((x,y): X- \texttt{sp} \le x  \le X+ \texttt{sp} , Y- \texttt{sp} \le y  \le Y+ \texttt{sp} , ||(R,G,B)-(r,g,b)||   \le \texttt{sr}\)
     *
     * where (R,G,B) and (r,g,b) are the vectors of color components at (X,Y) and (x,y), respectively
     * (though, the algorithm does not depend on the color space used, so any 3-component color space can
     * be used instead). Over the neighborhood the average spatial value (X',Y') and average color vector
     * (R',G',B') are found and they act as the neighborhood center on the next iteration:
     *
     * \((X,Y)~(X',Y'), (R,G,B)~(R',G',B').\)
     *
     * After the iterations over, the color components of the initial pixel (that is, the pixel from where
     * the iterations started) are set to the final value (average color at the last iteration):
     *
     * \(I(X,Y) &lt;- (R*,G*,B*)\)
     *
     * When maxLevel &gt; 0, the gaussian pyramid of maxLevel+1 levels is built, and the above procedure is
     * run on the smallest layer first. After that, the results are propagated to the larger layer and the
     * iterations are run again only on those pixels where the layer colors differ by more than sr from the
     * lower-resolution layer of the pyramid. That makes boundaries of color regions sharper. Note that the
     * results will be actually different from the ones obtained by running the meanshift procedure on the
     * whole original image (i.e. when maxLevel==0).
     *
     * @param src The source 8-bit, 3-channel image.
     * @param dst The destination image of the same format and the same size as the source.
     * @param sp The spatial window radius.
     * @param sr The color window radius.
     * @param maxLevel Maximum level of the pyramid for the segmentation.
     */
    public static void pyrMeanShiftFiltering(Mat src, Mat dst, double sp, double sr, int maxLevel) {
        pyrMeanShiftFiltering_1(src.nativeObj, dst.nativeObj, sp, sr, maxLevel);
    }

    /**
     * Performs initial step of meanshift segmentation of an image.
     *
     * The function implements the filtering stage of meanshift segmentation, that is, the output of the
     * function is the filtered "posterized" image with color gradients and fine-grain texture flattened.
     * At every pixel (X,Y) of the input image (or down-sized input image, see below) the function executes
     * meanshift iterations, that is, the pixel (X,Y) neighborhood in the joint space-color hyperspace is
     * considered:
     *
     * \((x,y): X- \texttt{sp} \le x  \le X+ \texttt{sp} , Y- \texttt{sp} \le y  \le Y+ \texttt{sp} , ||(R,G,B)-(r,g,b)||   \le \texttt{sr}\)
     *
     * where (R,G,B) and (r,g,b) are the vectors of color components at (X,Y) and (x,y), respectively
     * (though, the algorithm does not depend on the color space used, so any 3-component color space can
     * be used instead). Over the neighborhood the average spatial value (X',Y') and average color vector
     * (R',G',B') are found and they act as the neighborhood center on the next iteration:
     *
     * \((X,Y)~(X',Y'), (R,G,B)~(R',G',B').\)
     *
     * After the iterations over, the color components of the initial pixel (that is, the pixel from where
     * the iterations started) are set to the final value (average color at the last iteration):
     *
     * \(I(X,Y) &lt;- (R*,G*,B*)\)
     *
     * When maxLevel &gt; 0, the gaussian pyramid of maxLevel+1 levels is built, and the above procedure is
     * run on the smallest layer first. After that, the results are propagated to the larger layer and the
     * iterations are run again only on those pixels where the layer colors differ by more than sr from the
     * lower-resolution layer of the pyramid. That makes boundaries of color regions sharper. Note that the
     * results will be actually different from the ones obtained by running the meanshift procedure on the
     * whole original image (i.e. when maxLevel==0).
     *
     * @param src The source 8-bit, 3-channel image.
     * @param dst The destination image of the same format and the same size as the source.
     * @param sp The spatial window radius.
     * @param sr The color window radius.
     */
    public static void pyrMeanShiftFiltering(Mat src, Mat dst, double sp, double sr) {
        pyrMeanShiftFiltering_2(src.nativeObj, dst.nativeObj, sp, sr);
    }


    //
    // C++:  void cv::pyrUp(Mat src, Mat& dst, Size dstsize = Size(), int borderType = BORDER_DEFAULT)
    //

    /**
     * Upsamples an image and then blurs it.
     *
     * By default, size of the output image is computed as {@code Size(src.cols\*2, (src.rows\*2)}, but in any
     * case, the following conditions should be satisfied:
     *
     * \(\begin{array}{l} | \texttt{dstsize.width} -src.cols*2| \leq  ( \texttt{dstsize.width}   \mod  2)  \\ | \texttt{dstsize.height} -src.rows*2| \leq  ( \texttt{dstsize.height}   \mod  2) \end{array}\)
     *
     * The function performs the upsampling step of the Gaussian pyramid construction, though it can
     * actually be used to construct the Laplacian pyramid. First, it upsamples the source image by
     * injecting even zero rows and columns and then convolves the result with the same kernel as in
     * pyrDown multiplied by 4.
     *
     * @param src input image.
     * @param dst output image. It has the specified size and the same type as src .
     * @param dstsize size of the output image.
     * @param borderType Pixel extrapolation method, see #BorderTypes (only #BORDER_DEFAULT is supported)
     */
    public static void pyrUp(Mat src, Mat dst, Size dstsize, int borderType) {
        pyrUp_0(src.nativeObj, dst.nativeObj, dstsize.width, dstsize.height, borderType);
    }

    /**
     * Upsamples an image and then blurs it.
     *
     * By default, size of the output image is computed as {@code Size(src.cols\*2, (src.rows\*2)}, but in any
     * case, the following conditions should be satisfied:
     *
     * \(\begin{array}{l} | \texttt{dstsize.width} -src.cols*2| \leq  ( \texttt{dstsize.width}   \mod  2)  \\ | \texttt{dstsize.height} -src.rows*2| \leq  ( \texttt{dstsize.height}   \mod  2) \end{array}\)
     *
     * The function performs the upsampling step of the Gaussian pyramid construction, though it can
     * actually be used to construct the Laplacian pyramid. First, it upsamples the source image by
     * injecting even zero rows and columns and then convolves the result with the same kernel as in
     * pyrDown multiplied by 4.
     *
     * @param src input image.
     * @param dst output image. It has the specified size and the same type as src .
     * @param dstsize size of the output image.
     */
    public static void pyrUp(Mat src, Mat dst, Size dstsize) {
        pyrUp_1(src.nativeObj, dst.nativeObj, dstsize.width, dstsize.height);
    }

    /**
     * Upsamples an image and then blurs it.
     *
     * By default, size of the output image is computed as {@code Size(src.cols\*2, (src.rows\*2)}, but in any
     * case, the following conditions should be satisfied:
     *
     * \(\begin{array}{l} | \texttt{dstsize.width} -src.cols*2| \leq  ( \texttt{dstsize.width}   \mod  2)  \\ | \texttt{dstsize.height} -src.rows*2| \leq  ( \texttt{dstsize.height}   \mod  2) \end{array}\)
     *
     * The function performs the upsampling step of the Gaussian pyramid construction, though it can
     * actually be used to construct the Laplacian pyramid. First, it upsamples the source image by
     * injecting even zero rows and columns and then convolves the result with the same kernel as in
     * pyrDown multiplied by 4.
     *
     * @param src input image.
     * @param dst output image. It has the specified size and the same type as src .
     */
    public static void pyrUp(Mat src, Mat dst) {
        pyrUp_2(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::rectangle(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    //

    /**
     * Draws a simple, thick, or filled up-right rectangle.
     *
     * The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners
     * are pt1 and pt2.
     *
     * @param img Image.
     * @param pt1 Vertex of the rectangle.
     * @param pt2 Vertex of the rectangle opposite to pt1 .
     * @param color Rectangle color or brightness (grayscale image).
     * @param thickness Thickness of lines that make up the rectangle. Negative values, like #FILLED,
     * mean that the function has to draw a filled rectangle.
     * @param lineType Type of the line. See #LineTypes
     * @param shift Number of fractional bits in the point coordinates.
     */
    public static void rectangle(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int lineType, int shift) {
        rectangle_0(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, shift);
    }

    /**
     * Draws a simple, thick, or filled up-right rectangle.
     *
     * The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners
     * are pt1 and pt2.
     *
     * @param img Image.
     * @param pt1 Vertex of the rectangle.
     * @param pt2 Vertex of the rectangle opposite to pt1 .
     * @param color Rectangle color or brightness (grayscale image).
     * @param thickness Thickness of lines that make up the rectangle. Negative values, like #FILLED,
     * mean that the function has to draw a filled rectangle.
     * @param lineType Type of the line. See #LineTypes
     */
    public static void rectangle(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int lineType) {
        rectangle_1(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     * Draws a simple, thick, or filled up-right rectangle.
     *
     * The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners
     * are pt1 and pt2.
     *
     * @param img Image.
     * @param pt1 Vertex of the rectangle.
     * @param pt2 Vertex of the rectangle opposite to pt1 .
     * @param color Rectangle color or brightness (grayscale image).
     * @param thickness Thickness of lines that make up the rectangle. Negative values, like #FILLED,
     * mean that the function has to draw a filled rectangle.
     */
    public static void rectangle(Mat img, Point pt1, Point pt2, Scalar color, int thickness) {
        rectangle_2(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     * Draws a simple, thick, or filled up-right rectangle.
     *
     * The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners
     * are pt1 and pt2.
     *
     * @param img Image.
     * @param pt1 Vertex of the rectangle.
     * @param pt2 Vertex of the rectangle opposite to pt1 .
     * @param color Rectangle color or brightness (grayscale image).
     * mean that the function has to draw a filled rectangle.
     */
    public static void rectangle(Mat img, Point pt1, Point pt2, Scalar color) {
        rectangle_3(img.nativeObj, pt1.x, pt1.y, pt2.x, pt2.y, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::rectangle(Mat& img, Rect rec, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    //

    /**
     *
     *
     * use {@code rec} parameter as alternative specification of the drawn rectangle: `r.tl() and
     * r.br()-Point(1,1)` are opposite corners
     * @param img automatically generated
     * @param rec automatically generated
     * @param color automatically generated
     * @param thickness automatically generated
     * @param lineType automatically generated
     * @param shift automatically generated
     */
    public static void rectangle(Mat img, Rect rec, Scalar color, int thickness, int lineType, int shift) {
        rectangle_4(img.nativeObj, rec.x, rec.y, rec.width, rec.height, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType, shift);
    }

    /**
     *
     *
     * use {@code rec} parameter as alternative specification of the drawn rectangle: `r.tl() and
     * r.br()-Point(1,1)` are opposite corners
     * @param img automatically generated
     * @param rec automatically generated
     * @param color automatically generated
     * @param thickness automatically generated
     * @param lineType automatically generated
     */
    public static void rectangle(Mat img, Rect rec, Scalar color, int thickness, int lineType) {
        rectangle_5(img.nativeObj, rec.x, rec.y, rec.width, rec.height, color.val[0], color.val[1], color.val[2], color.val[3], thickness, lineType);
    }

    /**
     *
     *
     * use {@code rec} parameter as alternative specification of the drawn rectangle: `r.tl() and
     * r.br()-Point(1,1)` are opposite corners
     * @param img automatically generated
     * @param rec automatically generated
     * @param color automatically generated
     * @param thickness automatically generated
     */
    public static void rectangle(Mat img, Rect rec, Scalar color, int thickness) {
        rectangle_6(img.nativeObj, rec.x, rec.y, rec.width, rec.height, color.val[0], color.val[1], color.val[2], color.val[3], thickness);
    }

    /**
     *
     *
     * use {@code rec} parameter as alternative specification of the drawn rectangle: `r.tl() and
     * r.br()-Point(1,1)` are opposite corners
     * @param img automatically generated
     * @param rec automatically generated
     * @param color automatically generated
     */
    public static void rectangle(Mat img, Rect rec, Scalar color) {
        rectangle_7(img.nativeObj, rec.x, rec.y, rec.width, rec.height, color.val[0], color.val[1], color.val[2], color.val[3]);
    }


    //
    // C++:  void cv::remap(Mat src, Mat& dst, Mat map1, Mat map2, int interpolation, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
    //

    /**
     * Applies a generic geometrical transformation to an image.
     *
     * The function remap transforms the source image using the specified map:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))\)
     *
     * where values of pixels with non-integer coordinates are computed using one of available
     * interpolation methods. \(map_x\) and \(map_y\) can be encoded as separate floating-point maps
     * in \(map_1\) and \(map_2\) respectively, or interleaved floating-point maps of \((x,y)\) in
     * \(map_1\), or fixed-point maps created by using convertMaps. The reason you might want to
     * convert from floating to fixed-point representations of a map is that they can yield much faster
     * (\~2x) remapping operations. In the converted case, \(map_1\) contains pairs (cvFloor(x),
     * cvFloor(y)) and \(map_2\) contains indices in a table of interpolation coefficients.
     *
     * This function cannot operate in-place.
     *
     * @param src Source image.
     * @param dst Destination image. It has the same size as map1 and the same type as src .
     * @param map1 The first map of either (x,y) points or just x values having the type CV_16SC2 ,
     * CV_32FC1, or CV_32FC2. See convertMaps for details on converting a floating point
     * representation to fixed-point for speed.
     * @param map2 The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map
     * if map1 is (x,y) points), respectively.
     * @param interpolation Interpolation method (see #InterpolationFlags). The method #INTER_AREA is
     * not supported by this function.
     * @param borderMode Pixel extrapolation method (see #BorderTypes). When
     * borderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image that
     * corresponds to the "outliers" in the source image are not modified by the function.
     * @param borderValue Value used in case of a constant border. By default, it is 0.
     * <b>Note:</b>
     * Due to current implementation limitations the size of an input and output images should be less than 32767x32767.
     */
    public static void remap(Mat src, Mat dst, Mat map1, Mat map2, int interpolation, int borderMode, Scalar borderValue) {
        remap_0(src.nativeObj, dst.nativeObj, map1.nativeObj, map2.nativeObj, interpolation, borderMode, borderValue.val[0], borderValue.val[1], borderValue.val[2], borderValue.val[3]);
    }

    /**
     * Applies a generic geometrical transformation to an image.
     *
     * The function remap transforms the source image using the specified map:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))\)
     *
     * where values of pixels with non-integer coordinates are computed using one of available
     * interpolation methods. \(map_x\) and \(map_y\) can be encoded as separate floating-point maps
     * in \(map_1\) and \(map_2\) respectively, or interleaved floating-point maps of \((x,y)\) in
     * \(map_1\), or fixed-point maps created by using convertMaps. The reason you might want to
     * convert from floating to fixed-point representations of a map is that they can yield much faster
     * (\~2x) remapping operations. In the converted case, \(map_1\) contains pairs (cvFloor(x),
     * cvFloor(y)) and \(map_2\) contains indices in a table of interpolation coefficients.
     *
     * This function cannot operate in-place.
     *
     * @param src Source image.
     * @param dst Destination image. It has the same size as map1 and the same type as src .
     * @param map1 The first map of either (x,y) points or just x values having the type CV_16SC2 ,
     * CV_32FC1, or CV_32FC2. See convertMaps for details on converting a floating point
     * representation to fixed-point for speed.
     * @param map2 The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map
     * if map1 is (x,y) points), respectively.
     * @param interpolation Interpolation method (see #InterpolationFlags). The method #INTER_AREA is
     * not supported by this function.
     * @param borderMode Pixel extrapolation method (see #BorderTypes). When
     * borderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image that
     * corresponds to the "outliers" in the source image are not modified by the function.
     * <b>Note:</b>
     * Due to current implementation limitations the size of an input and output images should be less than 32767x32767.
     */
    public static void remap(Mat src, Mat dst, Mat map1, Mat map2, int interpolation, int borderMode) {
        remap_1(src.nativeObj, dst.nativeObj, map1.nativeObj, map2.nativeObj, interpolation, borderMode);
    }

    /**
     * Applies a generic geometrical transformation to an image.
     *
     * The function remap transforms the source image using the specified map:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))\)
     *
     * where values of pixels with non-integer coordinates are computed using one of available
     * interpolation methods. \(map_x\) and \(map_y\) can be encoded as separate floating-point maps
     * in \(map_1\) and \(map_2\) respectively, or interleaved floating-point maps of \((x,y)\) in
     * \(map_1\), or fixed-point maps created by using convertMaps. The reason you might want to
     * convert from floating to fixed-point representations of a map is that they can yield much faster
     * (\~2x) remapping operations. In the converted case, \(map_1\) contains pairs (cvFloor(x),
     * cvFloor(y)) and \(map_2\) contains indices in a table of interpolation coefficients.
     *
     * This function cannot operate in-place.
     *
     * @param src Source image.
     * @param dst Destination image. It has the same size as map1 and the same type as src .
     * @param map1 The first map of either (x,y) points or just x values having the type CV_16SC2 ,
     * CV_32FC1, or CV_32FC2. See convertMaps for details on converting a floating point
     * representation to fixed-point for speed.
     * @param map2 The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map
     * if map1 is (x,y) points), respectively.
     * @param interpolation Interpolation method (see #InterpolationFlags). The method #INTER_AREA is
     * not supported by this function.
     * borderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image that
     * corresponds to the "outliers" in the source image are not modified by the function.
     * <b>Note:</b>
     * Due to current implementation limitations the size of an input and output images should be less than 32767x32767.
     */
    public static void remap(Mat src, Mat dst, Mat map1, Mat map2, int interpolation) {
        remap_2(src.nativeObj, dst.nativeObj, map1.nativeObj, map2.nativeObj, interpolation);
    }


    //
    // C++:  void cv::resize(Mat src, Mat& dst, Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR)
    //

    /**
     * Resizes an image.
     *
     * The function resize resizes the image src down to or up to the specified size. Note that the
     * initial dst type or size are not taken into account. Instead, the size and type are derived from
     * the {@code src},{@code dsize},{@code fx}, and {@code fy}. If you want to resize src so that it fits the pre-created dst,
     * you may call the function as follows:
     * <code>
     *     // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
     *     resize(src, dst, dst.size(), 0, 0, interpolation);
     * </code>
     * If you want to decimate the image by factor of 2 in each direction, you can call the function this
     * way:
     * <code>
     *     // specify fx and fy and let the function compute the destination image size.
     *     resize(src, dst, Size(), 0.5, 0.5, interpolation);
     * </code>
     * To shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to
     * enlarge an image, it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR
     * (faster but still looks OK).
     *
     * @param src input image.
     * @param dst output image; it has the size dsize (when it is non-zero) or the size computed from
     * src.size(), fx, and fy; the type of dst is the same as of src.
     * @param dsize output image size; if it equals zero, it is computed as:
     *  \(\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\)
     *  Either dsize or both fx and fy must be non-zero.
     * @param fx scale factor along the horizontal axis; when it equals 0, it is computed as
     * \(\texttt{(double)dsize.width/src.cols}\)
     * @param fy scale factor along the vertical axis; when it equals 0, it is computed as
     * \(\texttt{(double)dsize.height/src.rows}\)
     * @param interpolation interpolation method, see #InterpolationFlags
     *
     * SEE:  warpAffine, warpPerspective, remap
     */
    public static void resize(Mat src, Mat dst, Size dsize, double fx, double fy, int interpolation) {
        resize_0(src.nativeObj, dst.nativeObj, dsize.width, dsize.height, fx, fy, interpolation);
    }

    /**
     * Resizes an image.
     *
     * The function resize resizes the image src down to or up to the specified size. Note that the
     * initial dst type or size are not taken into account. Instead, the size and type are derived from
     * the {@code src},{@code dsize},{@code fx}, and {@code fy}. If you want to resize src so that it fits the pre-created dst,
     * you may call the function as follows:
     * <code>
     *     // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
     *     resize(src, dst, dst.size(), 0, 0, interpolation);
     * </code>
     * If you want to decimate the image by factor of 2 in each direction, you can call the function this
     * way:
     * <code>
     *     // specify fx and fy and let the function compute the destination image size.
     *     resize(src, dst, Size(), 0.5, 0.5, interpolation);
     * </code>
     * To shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to
     * enlarge an image, it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR
     * (faster but still looks OK).
     *
     * @param src input image.
     * @param dst output image; it has the size dsize (when it is non-zero) or the size computed from
     * src.size(), fx, and fy; the type of dst is the same as of src.
     * @param dsize output image size; if it equals zero, it is computed as:
     *  \(\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\)
     *  Either dsize or both fx and fy must be non-zero.
     * @param fx scale factor along the horizontal axis; when it equals 0, it is computed as
     * \(\texttt{(double)dsize.width/src.cols}\)
     * @param fy scale factor along the vertical axis; when it equals 0, it is computed as
     * \(\texttt{(double)dsize.height/src.rows}\)
     *
     * SEE:  warpAffine, warpPerspective, remap
     */
    public static void resize(Mat src, Mat dst, Size dsize, double fx, double fy) {
        resize_1(src.nativeObj, dst.nativeObj, dsize.width, dsize.height, fx, fy);
    }

    /**
     * Resizes an image.
     *
     * The function resize resizes the image src down to or up to the specified size. Note that the
     * initial dst type or size are not taken into account. Instead, the size and type are derived from
     * the {@code src},{@code dsize},{@code fx}, and {@code fy}. If you want to resize src so that it fits the pre-created dst,
     * you may call the function as follows:
     * <code>
     *     // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
     *     resize(src, dst, dst.size(), 0, 0, interpolation);
     * </code>
     * If you want to decimate the image by factor of 2 in each direction, you can call the function this
     * way:
     * <code>
     *     // specify fx and fy and let the function compute the destination image size.
     *     resize(src, dst, Size(), 0.5, 0.5, interpolation);
     * </code>
     * To shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to
     * enlarge an image, it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR
     * (faster but still looks OK).
     *
     * @param src input image.
     * @param dst output image; it has the size dsize (when it is non-zero) or the size computed from
     * src.size(), fx, and fy; the type of dst is the same as of src.
     * @param dsize output image size; if it equals zero, it is computed as:
     *  \(\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\)
     *  Either dsize or both fx and fy must be non-zero.
     * @param fx scale factor along the horizontal axis; when it equals 0, it is computed as
     * \(\texttt{(double)dsize.width/src.cols}\)
     * \(\texttt{(double)dsize.height/src.rows}\)
     *
     * SEE:  warpAffine, warpPerspective, remap
     */
    public static void resize(Mat src, Mat dst, Size dsize, double fx) {
        resize_2(src.nativeObj, dst.nativeObj, dsize.width, dsize.height, fx);
    }

    /**
     * Resizes an image.
     *
     * The function resize resizes the image src down to or up to the specified size. Note that the
     * initial dst type or size are not taken into account. Instead, the size and type are derived from
     * the {@code src},{@code dsize},{@code fx}, and {@code fy}. If you want to resize src so that it fits the pre-created dst,
     * you may call the function as follows:
     * <code>
     *     // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
     *     resize(src, dst, dst.size(), 0, 0, interpolation);
     * </code>
     * If you want to decimate the image by factor of 2 in each direction, you can call the function this
     * way:
     * <code>
     *     // specify fx and fy and let the function compute the destination image size.
     *     resize(src, dst, Size(), 0.5, 0.5, interpolation);
     * </code>
     * To shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to
     * enlarge an image, it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR
     * (faster but still looks OK).
     *
     * @param src input image.
     * @param dst output image; it has the size dsize (when it is non-zero) or the size computed from
     * src.size(), fx, and fy; the type of dst is the same as of src.
     * @param dsize output image size; if it equals zero, it is computed as:
     *  \(\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\)
     *  Either dsize or both fx and fy must be non-zero.
     * \(\texttt{(double)dsize.width/src.cols}\)
     * \(\texttt{(double)dsize.height/src.rows}\)
     *
     * SEE:  warpAffine, warpPerspective, remap
     */
    public static void resize(Mat src, Mat dst, Size dsize) {
        resize_3(src.nativeObj, dst.nativeObj, dsize.width, dsize.height);
    }


    //
    // C++:  void cv::sepFilter2D(Mat src, Mat& dst, int ddepth, Mat kernelX, Mat kernelY, Point anchor = Point(-1,-1), double delta = 0, int borderType = BORDER_DEFAULT)
    //

    /**
     * Applies a separable linear filter to an image.
     *
     * The function applies a separable linear filter to the image. That is, first, every row of src is
     * filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D
     * kernel kernelY. The final result shifted by delta is stored in dst .
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Destination image depth, see REF: filter_depths "combinations"
     * @param kernelX Coefficients for filtering each row.
     * @param kernelY Coefficients for filtering each column.
     * @param anchor Anchor position within the kernel. The default value \((-1,-1)\) means that the anchor
     * is at the kernel center.
     * @param delta Value added to the filtered results before storing them.
     * @param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE:  filter2D, Sobel, GaussianBlur, boxFilter, blur
     */
    public static void sepFilter2D(Mat src, Mat dst, int ddepth, Mat kernelX, Mat kernelY, Point anchor, double delta, int borderType) {
        sepFilter2D_0(src.nativeObj, dst.nativeObj, ddepth, kernelX.nativeObj, kernelY.nativeObj, anchor.x, anchor.y, delta, borderType);
    }

    /**
     * Applies a separable linear filter to an image.
     *
     * The function applies a separable linear filter to the image. That is, first, every row of src is
     * filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D
     * kernel kernelY. The final result shifted by delta is stored in dst .
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Destination image depth, see REF: filter_depths "combinations"
     * @param kernelX Coefficients for filtering each row.
     * @param kernelY Coefficients for filtering each column.
     * @param anchor Anchor position within the kernel. The default value \((-1,-1)\) means that the anchor
     * is at the kernel center.
     * @param delta Value added to the filtered results before storing them.
     * SEE:  filter2D, Sobel, GaussianBlur, boxFilter, blur
     */
    public static void sepFilter2D(Mat src, Mat dst, int ddepth, Mat kernelX, Mat kernelY, Point anchor, double delta) {
        sepFilter2D_1(src.nativeObj, dst.nativeObj, ddepth, kernelX.nativeObj, kernelY.nativeObj, anchor.x, anchor.y, delta);
    }

    /**
     * Applies a separable linear filter to an image.
     *
     * The function applies a separable linear filter to the image. That is, first, every row of src is
     * filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D
     * kernel kernelY. The final result shifted by delta is stored in dst .
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Destination image depth, see REF: filter_depths "combinations"
     * @param kernelX Coefficients for filtering each row.
     * @param kernelY Coefficients for filtering each column.
     * @param anchor Anchor position within the kernel. The default value \((-1,-1)\) means that the anchor
     * is at the kernel center.
     * SEE:  filter2D, Sobel, GaussianBlur, boxFilter, blur
     */
    public static void sepFilter2D(Mat src, Mat dst, int ddepth, Mat kernelX, Mat kernelY, Point anchor) {
        sepFilter2D_2(src.nativeObj, dst.nativeObj, ddepth, kernelX.nativeObj, kernelY.nativeObj, anchor.x, anchor.y);
    }

    /**
     * Applies a separable linear filter to an image.
     *
     * The function applies a separable linear filter to the image. That is, first, every row of src is
     * filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D
     * kernel kernelY. The final result shifted by delta is stored in dst .
     *
     * @param src Source image.
     * @param dst Destination image of the same size and the same number of channels as src .
     * @param ddepth Destination image depth, see REF: filter_depths "combinations"
     * @param kernelX Coefficients for filtering each row.
     * @param kernelY Coefficients for filtering each column.
     * is at the kernel center.
     * SEE:  filter2D, Sobel, GaussianBlur, boxFilter, blur
     */
    public static void sepFilter2D(Mat src, Mat dst, int ddepth, Mat kernelX, Mat kernelY) {
        sepFilter2D_3(src.nativeObj, dst.nativeObj, ddepth, kernelX.nativeObj, kernelY.nativeObj);
    }


    //
    // C++:  void cv::spatialGradient(Mat src, Mat& dx, Mat& dy, int ksize = 3, int borderType = BORDER_DEFAULT)
    //

    /**
     * Calculates the first order image derivative in both x and y using a Sobel operator
     *
     * Equivalent to calling:
     *
     * <code>
     * Sobel( src, dx, CV_16SC1, 1, 0, 3 );
     * Sobel( src, dy, CV_16SC1, 0, 1, 3 );
     * </code>
     *
     * @param src input image.
     * @param dx output image with first-order derivative in x.
     * @param dy output image with first-order derivative in y.
     * @param ksize size of Sobel kernel. It must be 3.
     * @param borderType pixel extrapolation method, see #BorderTypes.
     *                   Only #BORDER_DEFAULT=#BORDER_REFLECT_101 and #BORDER_REPLICATE are supported.
     *
     * SEE: Sobel
     */
    public static void spatialGradient(Mat src, Mat dx, Mat dy, int ksize, int borderType) {
        spatialGradient_0(src.nativeObj, dx.nativeObj, dy.nativeObj, ksize, borderType);
    }

    /**
     * Calculates the first order image derivative in both x and y using a Sobel operator
     *
     * Equivalent to calling:
     *
     * <code>
     * Sobel( src, dx, CV_16SC1, 1, 0, 3 );
     * Sobel( src, dy, CV_16SC1, 0, 1, 3 );
     * </code>
     *
     * @param src input image.
     * @param dx output image with first-order derivative in x.
     * @param dy output image with first-order derivative in y.
     * @param ksize size of Sobel kernel. It must be 3.
     *                   Only #BORDER_DEFAULT=#BORDER_REFLECT_101 and #BORDER_REPLICATE are supported.
     *
     * SEE: Sobel
     */
    public static void spatialGradient(Mat src, Mat dx, Mat dy, int ksize) {
        spatialGradient_1(src.nativeObj, dx.nativeObj, dy.nativeObj, ksize);
    }

    /**
     * Calculates the first order image derivative in both x and y using a Sobel operator
     *
     * Equivalent to calling:
     *
     * <code>
     * Sobel( src, dx, CV_16SC1, 1, 0, 3 );
     * Sobel( src, dy, CV_16SC1, 0, 1, 3 );
     * </code>
     *
     * @param src input image.
     * @param dx output image with first-order derivative in x.
     * @param dy output image with first-order derivative in y.
     *                   Only #BORDER_DEFAULT=#BORDER_REFLECT_101 and #BORDER_REPLICATE are supported.
     *
     * SEE: Sobel
     */
    public static void spatialGradient(Mat src, Mat dx, Mat dy) {
        spatialGradient_2(src.nativeObj, dx.nativeObj, dy.nativeObj);
    }


    //
    // C++:  void cv::sqrBoxFilter(Mat src, Mat& dst, int ddepth, Size ksize, Point anchor = Point(-1, -1), bool normalize = true, int borderType = BORDER_DEFAULT)
    //

    /**
     * Calculates the normalized sum of squares of the pixel values overlapping the filter.
     *
     * For every pixel \( (x, y) \) in the source image, the function calculates the sum of squares of those neighboring
     * pixel values which overlap the filter placed over the pixel \( (x, y) \).
     *
     * The unnormalized square box filter can be useful in computing local image statistics such as the the local
     * variance and standard deviation around the neighborhood of a pixel.
     *
     * @param src input image
     * @param dst output image of the same size and type as _src
     * @param ddepth the output image depth (-1 to use src.depth())
     * @param ksize kernel size
     * @param anchor kernel anchor point. The default value of Point(-1, -1) denotes that the anchor is at the kernel
     * center.
     * @param normalize flag, specifying whether the kernel is to be normalized by it's area or not.
     * @param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes. #BORDER_WRAP is not supported.
     * SEE: boxFilter
     */
    public static void sqrBoxFilter(Mat src, Mat dst, int ddepth, Size ksize, Point anchor, boolean normalize, int borderType) {
        sqrBoxFilter_0(src.nativeObj, dst.nativeObj, ddepth, ksize.width, ksize.height, anchor.x, anchor.y, normalize, borderType);
    }

    /**
     * Calculates the normalized sum of squares of the pixel values overlapping the filter.
     *
     * For every pixel \( (x, y) \) in the source image, the function calculates the sum of squares of those neighboring
     * pixel values which overlap the filter placed over the pixel \( (x, y) \).
     *
     * The unnormalized square box filter can be useful in computing local image statistics such as the the local
     * variance and standard deviation around the neighborhood of a pixel.
     *
     * @param src input image
     * @param dst output image of the same size and type as _src
     * @param ddepth the output image depth (-1 to use src.depth())
     * @param ksize kernel size
     * @param anchor kernel anchor point. The default value of Point(-1, -1) denotes that the anchor is at the kernel
     * center.
     * @param normalize flag, specifying whether the kernel is to be normalized by it's area or not.
     * SEE: boxFilter
     */
    public static void sqrBoxFilter(Mat src, Mat dst, int ddepth, Size ksize, Point anchor, boolean normalize) {
        sqrBoxFilter_1(src.nativeObj, dst.nativeObj, ddepth, ksize.width, ksize.height, anchor.x, anchor.y, normalize);
    }

    /**
     * Calculates the normalized sum of squares of the pixel values overlapping the filter.
     *
     * For every pixel \( (x, y) \) in the source image, the function calculates the sum of squares of those neighboring
     * pixel values which overlap the filter placed over the pixel \( (x, y) \).
     *
     * The unnormalized square box filter can be useful in computing local image statistics such as the the local
     * variance and standard deviation around the neighborhood of a pixel.
     *
     * @param src input image
     * @param dst output image of the same size and type as _src
     * @param ddepth the output image depth (-1 to use src.depth())
     * @param ksize kernel size
     * @param anchor kernel anchor point. The default value of Point(-1, -1) denotes that the anchor is at the kernel
     * center.
     * SEE: boxFilter
     */
    public static void sqrBoxFilter(Mat src, Mat dst, int ddepth, Size ksize, Point anchor) {
        sqrBoxFilter_2(src.nativeObj, dst.nativeObj, ddepth, ksize.width, ksize.height, anchor.x, anchor.y);
    }

    /**
     * Calculates the normalized sum of squares of the pixel values overlapping the filter.
     *
     * For every pixel \( (x, y) \) in the source image, the function calculates the sum of squares of those neighboring
     * pixel values which overlap the filter placed over the pixel \( (x, y) \).
     *
     * The unnormalized square box filter can be useful in computing local image statistics such as the the local
     * variance and standard deviation around the neighborhood of a pixel.
     *
     * @param src input image
     * @param dst output image of the same size and type as _src
     * @param ddepth the output image depth (-1 to use src.depth())
     * @param ksize kernel size
     * center.
     * SEE: boxFilter
     */
    public static void sqrBoxFilter(Mat src, Mat dst, int ddepth, Size ksize) {
        sqrBoxFilter_3(src.nativeObj, dst.nativeObj, ddepth, ksize.width, ksize.height);
    }


    //
    // C++:  void cv::warpAffine(Mat src, Mat& dst, Mat M, Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
    //

    /**
     * Applies an affine transformation to an image.
     *
     * The function warpAffine transforms the source image using the specified matrix:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\)
     *
     * when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted
     * with #invertAffineTransform and then put in the formula above instead of M. The function cannot
     * operate in-place.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \(2\times 3\) transformation matrix.
     * @param dsize size of the output image.
     * @param flags combination of interpolation methods (see #InterpolationFlags) and the optional
     * flag #WARP_INVERSE_MAP that means that M is the inverse transformation (
     * \(\texttt{dst}\rightarrow\texttt{src}\) ).
     * @param borderMode pixel extrapolation method (see #BorderTypes); when
     * borderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to
     * the "outliers" in the source image are not modified by the function.
     * @param borderValue value used in case of a constant border; by default, it is 0.
     *
     * SEE:  warpPerspective, resize, remap, getRectSubPix, transform
     */
    public static void warpAffine(Mat src, Mat dst, Mat M, Size dsize, int flags, int borderMode, Scalar borderValue) {
        warpAffine_0(src.nativeObj, dst.nativeObj, M.nativeObj, dsize.width, dsize.height, flags, borderMode, borderValue.val[0], borderValue.val[1], borderValue.val[2], borderValue.val[3]);
    }

    /**
     * Applies an affine transformation to an image.
     *
     * The function warpAffine transforms the source image using the specified matrix:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\)
     *
     * when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted
     * with #invertAffineTransform and then put in the formula above instead of M. The function cannot
     * operate in-place.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \(2\times 3\) transformation matrix.
     * @param dsize size of the output image.
     * @param flags combination of interpolation methods (see #InterpolationFlags) and the optional
     * flag #WARP_INVERSE_MAP that means that M is the inverse transformation (
     * \(\texttt{dst}\rightarrow\texttt{src}\) ).
     * @param borderMode pixel extrapolation method (see #BorderTypes); when
     * borderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to
     * the "outliers" in the source image are not modified by the function.
     *
     * SEE:  warpPerspective, resize, remap, getRectSubPix, transform
     */
    public static void warpAffine(Mat src, Mat dst, Mat M, Size dsize, int flags, int borderMode) {
        warpAffine_1(src.nativeObj, dst.nativeObj, M.nativeObj, dsize.width, dsize.height, flags, borderMode);
    }

    /**
     * Applies an affine transformation to an image.
     *
     * The function warpAffine transforms the source image using the specified matrix:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\)
     *
     * when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted
     * with #invertAffineTransform and then put in the formula above instead of M. The function cannot
     * operate in-place.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \(2\times 3\) transformation matrix.
     * @param dsize size of the output image.
     * @param flags combination of interpolation methods (see #InterpolationFlags) and the optional
     * flag #WARP_INVERSE_MAP that means that M is the inverse transformation (
     * \(\texttt{dst}\rightarrow\texttt{src}\) ).
     * borderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to
     * the "outliers" in the source image are not modified by the function.
     *
     * SEE:  warpPerspective, resize, remap, getRectSubPix, transform
     */
    public static void warpAffine(Mat src, Mat dst, Mat M, Size dsize, int flags) {
        warpAffine_2(src.nativeObj, dst.nativeObj, M.nativeObj, dsize.width, dsize.height, flags);
    }

    /**
     * Applies an affine transformation to an image.
     *
     * The function warpAffine transforms the source image using the specified matrix:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\)
     *
     * when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted
     * with #invertAffineTransform and then put in the formula above instead of M. The function cannot
     * operate in-place.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \(2\times 3\) transformation matrix.
     * @param dsize size of the output image.
     * flag #WARP_INVERSE_MAP that means that M is the inverse transformation (
     * \(\texttt{dst}\rightarrow\texttt{src}\) ).
     * borderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to
     * the "outliers" in the source image are not modified by the function.
     *
     * SEE:  warpPerspective, resize, remap, getRectSubPix, transform
     */
    public static void warpAffine(Mat src, Mat dst, Mat M, Size dsize) {
        warpAffine_3(src.nativeObj, dst.nativeObj, M.nativeObj, dsize.width, dsize.height);
    }


    //
    // C++:  void cv::warpPerspective(Mat src, Mat& dst, Mat M, Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
    //

    /**
     * Applies a perspective transformation to an image.
     *
     * The function warpPerspective transforms the source image using the specified matrix:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
     *      \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )\)
     *
     * when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert
     * and then put in the formula above instead of M. The function cannot operate in-place.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \(3\times 3\) transformation matrix.
     * @param dsize size of the output image.
     * @param flags combination of interpolation methods (#INTER_LINEAR or #INTER_NEAREST) and the
     * optional flag #WARP_INVERSE_MAP, that sets M as the inverse transformation (
     * \(\texttt{dst}\rightarrow\texttt{src}\) ).
     * @param borderMode pixel extrapolation method (#BORDER_CONSTANT or #BORDER_REPLICATE).
     * @param borderValue value used in case of a constant border; by default, it equals 0.
     *
     * SEE:  warpAffine, resize, remap, getRectSubPix, perspectiveTransform
     */
    public static void warpPerspective(Mat src, Mat dst, Mat M, Size dsize, int flags, int borderMode, Scalar borderValue) {
        warpPerspective_0(src.nativeObj, dst.nativeObj, M.nativeObj, dsize.width, dsize.height, flags, borderMode, borderValue.val[0], borderValue.val[1], borderValue.val[2], borderValue.val[3]);
    }

    /**
     * Applies a perspective transformation to an image.
     *
     * The function warpPerspective transforms the source image using the specified matrix:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
     *      \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )\)
     *
     * when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert
     * and then put in the formula above instead of M. The function cannot operate in-place.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \(3\times 3\) transformation matrix.
     * @param dsize size of the output image.
     * @param flags combination of interpolation methods (#INTER_LINEAR or #INTER_NEAREST) and the
     * optional flag #WARP_INVERSE_MAP, that sets M as the inverse transformation (
     * \(\texttt{dst}\rightarrow\texttt{src}\) ).
     * @param borderMode pixel extrapolation method (#BORDER_CONSTANT or #BORDER_REPLICATE).
     *
     * SEE:  warpAffine, resize, remap, getRectSubPix, perspectiveTransform
     */
    public static void warpPerspective(Mat src, Mat dst, Mat M, Size dsize, int flags, int borderMode) {
        warpPerspective_1(src.nativeObj, dst.nativeObj, M.nativeObj, dsize.width, dsize.height, flags, borderMode);
    }

    /**
     * Applies a perspective transformation to an image.
     *
     * The function warpPerspective transforms the source image using the specified matrix:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
     *      \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )\)
     *
     * when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert
     * and then put in the formula above instead of M. The function cannot operate in-place.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \(3\times 3\) transformation matrix.
     * @param dsize size of the output image.
     * @param flags combination of interpolation methods (#INTER_LINEAR or #INTER_NEAREST) and the
     * optional flag #WARP_INVERSE_MAP, that sets M as the inverse transformation (
     * \(\texttt{dst}\rightarrow\texttt{src}\) ).
     *
     * SEE:  warpAffine, resize, remap, getRectSubPix, perspectiveTransform
     */
    public static void warpPerspective(Mat src, Mat dst, Mat M, Size dsize, int flags) {
        warpPerspective_2(src.nativeObj, dst.nativeObj, M.nativeObj, dsize.width, dsize.height, flags);
    }

    /**
     * Applies a perspective transformation to an image.
     *
     * The function warpPerspective transforms the source image using the specified matrix:
     *
     * \(\texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
     *      \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )\)
     *
     * when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert
     * and then put in the formula above instead of M. The function cannot operate in-place.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \(3\times 3\) transformation matrix.
     * @param dsize size of the output image.
     * optional flag #WARP_INVERSE_MAP, that sets M as the inverse transformation (
     * \(\texttt{dst}\rightarrow\texttt{src}\) ).
     *
     * SEE:  warpAffine, resize, remap, getRectSubPix, perspectiveTransform
     */
    public static void warpPerspective(Mat src, Mat dst, Mat M, Size dsize) {
        warpPerspective_3(src.nativeObj, dst.nativeObj, M.nativeObj, dsize.width, dsize.height);
    }


    //
    // C++:  void cv::warpPolar(Mat src, Mat& dst, Size dsize, Point2f center, double maxRadius, int flags)
    //

    /**
     * Remaps an image to polar or semilog-polar coordinates space
     *
     *  polar_remaps_reference_image
     * ![Polar remaps reference](pics/polar_remap_doc.png)
     *
     * Transform the source image using the following transformation:
     * \(
     * dst(\rho , \phi ) = src(x,y)
     * \)
     *
     * where
     * \(
     * \begin{array}{l}
     * \vec{I} = (x - center.x, \;y - center.y) \\
     * \phi = Kangle \cdot \texttt{angle} (\vec{I}) \\
     * \rho = \left\{\begin{matrix}
     * Klin \cdot \texttt{magnitude} (\vec{I}) &amp; default \\
     * Klog \cdot log_e(\texttt{magnitude} (\vec{I})) &amp; if \; semilog \\
     * \end{matrix}\right.
     * \end{array}
     * \)
     *
     * and
     * \(
     * \begin{array}{l}
     * Kangle = dsize.height / 2\Pi \\
     * Klin = dsize.width / maxRadius \\
     * Klog = dsize.width / log_e(maxRadius) \\
     * \end{array}
     * \)
     *
     *
     * \par Linear vs semilog mapping
     *
     * Polar mapping can be linear or semi-log. Add one of #WarpPolarMode to {@code flags} to specify the polar mapping mode.
     *
     * Linear is the default mode.
     *
     * The semilog mapping emulates the human "foveal" vision that permit very high acuity on the line of sight (central vision)
     * in contrast to peripheral vision where acuity is minor.
     *
     * \par Option on {@code dsize}:
     *
     * <ul>
     *   <li>
     *  if both values in {@code dsize &lt;=0 } (default),
     * the destination image will have (almost) same area of source bounding circle:
     * \(\begin{array}{l}
     * dsize.area  \leftarrow (maxRadius^2 \cdot \Pi) \\
     * dsize.width = \texttt{cvRound}(maxRadius) \\
     * dsize.height = \texttt{cvRound}(maxRadius \cdot \Pi) \\
     * \end{array}\)
     *   </li>
     * </ul>
     *
     *
     * <ul>
     *   <li>
     *  if only {@code dsize.height &lt;= 0},
     * the destination image area will be proportional to the bounding circle area but scaled by {@code Kx * Kx}:
     * \(\begin{array}{l}
     * dsize.height = \texttt{cvRound}(dsize.width \cdot \Pi) \\
     * \end{array}
     * \)
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *  if both values in {@code dsize &gt; 0 },
     * the destination image will have the given size therefore the area of the bounding circle will be scaled to {@code dsize}.
     *   </li>
     * </ul>
     *
     *
     * \par Reverse mapping
     *
     * You can get reverse mapping adding #WARP_INVERSE_MAP to {@code flags}
     * \snippet polar_transforms.cpp InverseMap
     *
     * In addiction, to calculate the original coordinate from a polar mapped coordinate \((rho, phi)-&gt;(x, y)\):
     * \snippet polar_transforms.cpp InverseCoordinate
     *
     * @param src Source image.
     * @param dst Destination image. It will have same type as src.
     * @param dsize The destination image size (see description for valid options).
     * @param center The transformation center.
     * @param maxRadius The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.
     * @param flags A combination of interpolation methods, #InterpolationFlags + #WarpPolarMode.
     * <ul>
     *   <li>
     *              Add #WARP_POLAR_LINEAR to select linear polar mapping (default)
     *   </li>
     *   <li>
     *              Add #WARP_POLAR_LOG to select semilog polar mapping
     *   </li>
     *   <li>
     *              Add #WARP_INVERSE_MAP for reverse mapping.
     *   </li>
     * </ul>
     * <b>Note:</b>
     * <ul>
     *   <li>
     *   The function can not operate in-place.
     *   </li>
     *   <li>
     *   To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.
     *   </li>
     *   <li>
     *   This function uses #remap. Due to current implementation limitations the size of an input and output images should be less than 32767x32767.
     *   </li>
     * </ul>
     *
     * SEE: cv::remap
     */
    public static void warpPolar(Mat src, Mat dst, Size dsize, Point center, double maxRadius, int flags) {
        warpPolar_0(src.nativeObj, dst.nativeObj, dsize.width, dsize.height, center.x, center.y, maxRadius, flags);
    }


    //
    // C++:  void cv::watershed(Mat image, Mat& markers)
    //

    /**
     * Performs a marker-based image segmentation using the watershed algorithm.
     *
     * The function implements one of the variants of watershed, non-parametric marker-based segmentation
     * algorithm, described in CITE: Meyer92 .
     *
     * Before passing the image to the function, you have to roughly outline the desired regions in the
     * image markers with positive (&gt;0) indices. So, every region is represented as one or more connected
     * components with the pixel values 1, 2, 3, and so on. Such markers can be retrieved from a binary
     * mask using #findContours and #drawContours (see the watershed.cpp demo). The markers are "seeds" of
     * the future image regions. All the other pixels in markers , whose relation to the outlined regions
     * is not known and should be defined by the algorithm, should be set to 0's. In the function output,
     * each pixel in markers is set to a value of the "seed" components or to -1 at boundaries between the
     * regions.
     *
     * <b>Note:</b> Any two neighbor connected components are not necessarily separated by a watershed boundary
     * (-1's pixels); for example, they can touch each other in the initial marker image passed to the
     * function.
     *
     * @param image Input 8-bit 3-channel image.
     * @param markers Input/output 32-bit single-channel image (map) of markers. It should have the same
     * size as image .
     *
     * SEE: findContours
     *
     *  imgproc_misc
     */
    public static void watershed(Mat image, Mat markers) {
        watershed_0(image.nativeObj, markers.nativeObj);
    }



// C++: Size getTextSize(const String& text, int fontFace, double fontScale, int thickness, int* baseLine);
//javadoc:getTextSize(text, fontFace, fontScale, thickness, baseLine)
public static Size getTextSize(String text, int fontFace, double fontScale, int thickness, int[] baseLine) {
    if(baseLine != null && baseLine.length != 1)
        throw new java.lang.IllegalArgumentException("'baseLine' must be 'int[1]' or 'null'.");
    Size retVal = new Size(n_getTextSize(text, fontFace, fontScale, thickness, baseLine));
    return retVal;
}




    // C++:  Mat cv::getAffineTransform(vector_Point2f src, vector_Point2f dst)
    private static native long getAffineTransform_0(long src_mat_nativeObj, long dst_mat_nativeObj);

    // C++:  Mat cv::getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi = CV_PI*0.5, int ktype = CV_64F)
    private static native long getGaborKernel_0(double ksize_width, double ksize_height, double sigma, double theta, double lambd, double gamma, double psi, int ktype);
    private static native long getGaborKernel_1(double ksize_width, double ksize_height, double sigma, double theta, double lambd, double gamma, double psi);
    private static native long getGaborKernel_2(double ksize_width, double ksize_height, double sigma, double theta, double lambd, double gamma);

    // C++:  Mat cv::getGaussianKernel(int ksize, double sigma, int ktype = CV_64F)
    private static native long getGaussianKernel_0(int ksize, double sigma, int ktype);
    private static native long getGaussianKernel_1(int ksize, double sigma);

    // C++:  Mat cv::getPerspectiveTransform(Mat src, Mat dst, int solveMethod = DECOMP_LU)
    private static native long getPerspectiveTransform_0(long src_nativeObj, long dst_nativeObj, int solveMethod);
    private static native long getPerspectiveTransform_1(long src_nativeObj, long dst_nativeObj);

    // C++:  Mat cv::getRotationMatrix2D(Point2f center, double angle, double scale)
    private static native long getRotationMatrix2D_0(double center_x, double center_y, double angle, double scale);

    // C++:  Mat cv::getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1))
    private static native long getStructuringElement_0(int shape, double ksize_width, double ksize_height, double anchor_x, double anchor_y);
    private static native long getStructuringElement_1(int shape, double ksize_width, double ksize_height);

    // C++:  Moments cv::moments(Mat array, bool binaryImage = false)
    private static native double[] moments_0(long array_nativeObj, boolean binaryImage);
    private static native double[] moments_1(long array_nativeObj);

    // C++:  Point2d cv::phaseCorrelate(Mat src1, Mat src2, Mat window = Mat(), double* response = 0)
    private static native double[] phaseCorrelate_0(long src1_nativeObj, long src2_nativeObj, long window_nativeObj, double[] response_out);
    private static native double[] phaseCorrelate_1(long src1_nativeObj, long src2_nativeObj, long window_nativeObj);
    private static native double[] phaseCorrelate_2(long src1_nativeObj, long src2_nativeObj);

    // C++:  Ptr_CLAHE cv::createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8))
    private static native long createCLAHE_0(double clipLimit, double tileGridSize_width, double tileGridSize_height);
    private static native long createCLAHE_1(double clipLimit);
    private static native long createCLAHE_2();

    // C++:  Ptr_GeneralizedHoughBallard cv::createGeneralizedHoughBallard()
    private static native long createGeneralizedHoughBallard_0();

    // C++:  Ptr_GeneralizedHoughGuil cv::createGeneralizedHoughGuil()
    private static native long createGeneralizedHoughGuil_0();

    // C++:  Ptr_LineSegmentDetector cv::createLineSegmentDetector(int _refine = LSD_REFINE_STD, double _scale = 0.8, double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5, double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024)
    private static native long createLineSegmentDetector_0(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th, double _log_eps, double _density_th, int _n_bins);
    private static native long createLineSegmentDetector_1(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th, double _log_eps, double _density_th);
    private static native long createLineSegmentDetector_2(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th, double _log_eps);
    private static native long createLineSegmentDetector_3(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th);
    private static native long createLineSegmentDetector_4(int _refine, double _scale, double _sigma_scale, double _quant);
    private static native long createLineSegmentDetector_5(int _refine, double _scale, double _sigma_scale);
    private static native long createLineSegmentDetector_6(int _refine, double _scale);
    private static native long createLineSegmentDetector_7(int _refine);
    private static native long createLineSegmentDetector_8();

    // C++:  Rect cv::boundingRect(Mat array)
    private static native double[] boundingRect_0(long array_nativeObj);

    // C++:  RotatedRect cv::fitEllipse(vector_Point2f points)
    private static native double[] fitEllipse_0(long points_mat_nativeObj);

    // C++:  RotatedRect cv::fitEllipseAMS(Mat points)
    private static native double[] fitEllipseAMS_0(long points_nativeObj);

    // C++:  RotatedRect cv::fitEllipseDirect(Mat points)
    private static native double[] fitEllipseDirect_0(long points_nativeObj);

    // C++:  RotatedRect cv::minAreaRect(vector_Point2f points)
    private static native double[] minAreaRect_0(long points_mat_nativeObj);

    // C++:  bool cv::clipLine(Rect imgRect, Point& pt1, Point& pt2)
    private static native boolean clipLine_0(int imgRect_x, int imgRect_y, int imgRect_width, int imgRect_height, double pt1_x, double pt1_y, double[] pt1_out, double pt2_x, double pt2_y, double[] pt2_out);

    // C++:  bool cv::isContourConvex(vector_Point contour)
    private static native boolean isContourConvex_0(long contour_mat_nativeObj);

    // C++:  double cv::arcLength(vector_Point2f curve, bool closed)
    private static native double arcLength_0(long curve_mat_nativeObj, boolean closed);

    // C++:  double cv::compareHist(Mat H1, Mat H2, int method)
    private static native double compareHist_0(long H1_nativeObj, long H2_nativeObj, int method);

    // C++:  double cv::contourArea(Mat contour, bool oriented = false)
    private static native double contourArea_0(long contour_nativeObj, boolean oriented);
    private static native double contourArea_1(long contour_nativeObj);

    // C++:  double cv::getFontScaleFromHeight(int fontFace, int pixelHeight, int thickness = 1)
    private static native double getFontScaleFromHeight_0(int fontFace, int pixelHeight, int thickness);
    private static native double getFontScaleFromHeight_1(int fontFace, int pixelHeight);

    // C++:  double cv::matchShapes(Mat contour1, Mat contour2, int method, double parameter)
    private static native double matchShapes_0(long contour1_nativeObj, long contour2_nativeObj, int method, double parameter);

    // C++:  double cv::minEnclosingTriangle(Mat points, Mat& triangle)
    private static native double minEnclosingTriangle_0(long points_nativeObj, long triangle_nativeObj);

    // C++:  double cv::pointPolygonTest(vector_Point2f contour, Point2f pt, bool measureDist)
    private static native double pointPolygonTest_0(long contour_mat_nativeObj, double pt_x, double pt_y, boolean measureDist);

    // C++:  double cv::threshold(Mat src, Mat& dst, double thresh, double maxval, int type)
    private static native double threshold_0(long src_nativeObj, long dst_nativeObj, double thresh, double maxval, int type);

    // C++:  float cv::intersectConvexConvex(Mat _p1, Mat _p2, Mat& _p12, bool handleNested = true)
    private static native float intersectConvexConvex_0(long _p1_nativeObj, long _p2_nativeObj, long _p12_nativeObj, boolean handleNested);
    private static native float intersectConvexConvex_1(long _p1_nativeObj, long _p2_nativeObj, long _p12_nativeObj);

    // C++:  float cv::wrapperEMD(Mat signature1, Mat signature2, int distType, Mat cost = Mat(), Ptr_float& lowerBound = Ptr<float>(), Mat& flow = Mat())
    private static native float EMD_0(long signature1_nativeObj, long signature2_nativeObj, int distType, long cost_nativeObj, long flow_nativeObj);
    private static native float EMD_1(long signature1_nativeObj, long signature2_nativeObj, int distType, long cost_nativeObj);
    private static native float EMD_3(long signature1_nativeObj, long signature2_nativeObj, int distType);

    // C++:  int cv::connectedComponents(Mat image, Mat& labels, int connectivity, int ltype, int ccltype)
    private static native int connectedComponentsWithAlgorithm_0(long image_nativeObj, long labels_nativeObj, int connectivity, int ltype, int ccltype);

    // C++:  int cv::connectedComponents(Mat image, Mat& labels, int connectivity = 8, int ltype = CV_32S)
    private static native int connectedComponents_0(long image_nativeObj, long labels_nativeObj, int connectivity, int ltype);
    private static native int connectedComponents_1(long image_nativeObj, long labels_nativeObj, int connectivity);
    private static native int connectedComponents_2(long image_nativeObj, long labels_nativeObj);

    // C++:  int cv::connectedComponentsWithStats(Mat image, Mat& labels, Mat& stats, Mat& centroids, int connectivity, int ltype, int ccltype)
    private static native int connectedComponentsWithStatsWithAlgorithm_0(long image_nativeObj, long labels_nativeObj, long stats_nativeObj, long centroids_nativeObj, int connectivity, int ltype, int ccltype);

    // C++:  int cv::connectedComponentsWithStats(Mat image, Mat& labels, Mat& stats, Mat& centroids, int connectivity = 8, int ltype = CV_32S)
    private static native int connectedComponentsWithStats_0(long image_nativeObj, long labels_nativeObj, long stats_nativeObj, long centroids_nativeObj, int connectivity, int ltype);
    private static native int connectedComponentsWithStats_1(long image_nativeObj, long labels_nativeObj, long stats_nativeObj, long centroids_nativeObj, int connectivity);
    private static native int connectedComponentsWithStats_2(long image_nativeObj, long labels_nativeObj, long stats_nativeObj, long centroids_nativeObj);

    // C++:  int cv::floodFill(Mat& image, Mat& mask, Point seedPoint, Scalar newVal, Rect* rect = 0, Scalar loDiff = Scalar(), Scalar upDiff = Scalar(), int flags = 4)
    private static native int floodFill_0(long image_nativeObj, long mask_nativeObj, double seedPoint_x, double seedPoint_y, double newVal_val0, double newVal_val1, double newVal_val2, double newVal_val3, double[] rect_out, double loDiff_val0, double loDiff_val1, double loDiff_val2, double loDiff_val3, double upDiff_val0, double upDiff_val1, double upDiff_val2, double upDiff_val3, int flags);
    private static native int floodFill_1(long image_nativeObj, long mask_nativeObj, double seedPoint_x, double seedPoint_y, double newVal_val0, double newVal_val1, double newVal_val2, double newVal_val3, double[] rect_out, double loDiff_val0, double loDiff_val1, double loDiff_val2, double loDiff_val3, double upDiff_val0, double upDiff_val1, double upDiff_val2, double upDiff_val3);
    private static native int floodFill_2(long image_nativeObj, long mask_nativeObj, double seedPoint_x, double seedPoint_y, double newVal_val0, double newVal_val1, double newVal_val2, double newVal_val3, double[] rect_out, double loDiff_val0, double loDiff_val1, double loDiff_val2, double loDiff_val3);
    private static native int floodFill_3(long image_nativeObj, long mask_nativeObj, double seedPoint_x, double seedPoint_y, double newVal_val0, double newVal_val1, double newVal_val2, double newVal_val3, double[] rect_out);
    private static native int floodFill_4(long image_nativeObj, long mask_nativeObj, double seedPoint_x, double seedPoint_y, double newVal_val0, double newVal_val1, double newVal_val2, double newVal_val3);

    // C++:  int cv::rotatedRectangleIntersection(RotatedRect rect1, RotatedRect rect2, Mat& intersectingRegion)
    private static native int rotatedRectangleIntersection_0(double rect1_center_x, double rect1_center_y, double rect1_size_width, double rect1_size_height, double rect1_angle, double rect2_center_x, double rect2_center_y, double rect2_size_width, double rect2_size_height, double rect2_angle, long intersectingRegion_nativeObj);

    // C++:  void cv::Canny(Mat dx, Mat dy, Mat& edges, double threshold1, double threshold2, bool L2gradient = false)
    private static native void Canny_0(long dx_nativeObj, long dy_nativeObj, long edges_nativeObj, double threshold1, double threshold2, boolean L2gradient);
    private static native void Canny_1(long dx_nativeObj, long dy_nativeObj, long edges_nativeObj, double threshold1, double threshold2);

    // C++:  void cv::Canny(Mat image, Mat& edges, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false)
    private static native void Canny_2(long image_nativeObj, long edges_nativeObj, double threshold1, double threshold2, int apertureSize, boolean L2gradient);
    private static native void Canny_3(long image_nativeObj, long edges_nativeObj, double threshold1, double threshold2, int apertureSize);
    private static native void Canny_4(long image_nativeObj, long edges_nativeObj, double threshold1, double threshold2);

    // C++:  void cv::GaussianBlur(Mat src, Mat& dst, Size ksize, double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT)
    private static native void GaussianBlur_0(long src_nativeObj, long dst_nativeObj, double ksize_width, double ksize_height, double sigmaX, double sigmaY, int borderType);
    private static native void GaussianBlur_1(long src_nativeObj, long dst_nativeObj, double ksize_width, double ksize_height, double sigmaX, double sigmaY);
    private static native void GaussianBlur_2(long src_nativeObj, long dst_nativeObj, double ksize_width, double ksize_height, double sigmaX);

    // C++:  void cv::HoughCircles(Mat image, Mat& circles, int method, double dp, double minDist, double param1 = 100, double param2 = 100, int minRadius = 0, int maxRadius = 0)
    private static native void HoughCircles_0(long image_nativeObj, long circles_nativeObj, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius);
    private static native void HoughCircles_1(long image_nativeObj, long circles_nativeObj, int method, double dp, double minDist, double param1, double param2, int minRadius);
    private static native void HoughCircles_2(long image_nativeObj, long circles_nativeObj, int method, double dp, double minDist, double param1, double param2);
    private static native void HoughCircles_3(long image_nativeObj, long circles_nativeObj, int method, double dp, double minDist, double param1);
    private static native void HoughCircles_4(long image_nativeObj, long circles_nativeObj, int method, double dp, double minDist);

    // C++:  void cv::HoughLines(Mat image, Mat& lines, double rho, double theta, int threshold, double srn = 0, double stn = 0, double min_theta = 0, double max_theta = CV_PI)
    private static native void HoughLines_0(long image_nativeObj, long lines_nativeObj, double rho, double theta, int threshold, double srn, double stn, double min_theta, double max_theta);
    private static native void HoughLines_1(long image_nativeObj, long lines_nativeObj, double rho, double theta, int threshold, double srn, double stn, double min_theta);
    private static native void HoughLines_2(long image_nativeObj, long lines_nativeObj, double rho, double theta, int threshold, double srn, double stn);
    private static native void HoughLines_3(long image_nativeObj, long lines_nativeObj, double rho, double theta, int threshold, double srn);
    private static native void HoughLines_4(long image_nativeObj, long lines_nativeObj, double rho, double theta, int threshold);

    // C++:  void cv::HoughLinesP(Mat image, Mat& lines, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0)
    private static native void HoughLinesP_0(long image_nativeObj, long lines_nativeObj, double rho, double theta, int threshold, double minLineLength, double maxLineGap);
    private static native void HoughLinesP_1(long image_nativeObj, long lines_nativeObj, double rho, double theta, int threshold, double minLineLength);
    private static native void HoughLinesP_2(long image_nativeObj, long lines_nativeObj, double rho, double theta, int threshold);

    // C++:  void cv::HoughLinesPointSet(Mat _point, Mat& _lines, int lines_max, int threshold, double min_rho, double max_rho, double rho_step, double min_theta, double max_theta, double theta_step)
    private static native void HoughLinesPointSet_0(long _point_nativeObj, long _lines_nativeObj, int lines_max, int threshold, double min_rho, double max_rho, double rho_step, double min_theta, double max_theta, double theta_step);

    // C++:  void cv::HuMoments(Moments m, Mat& hu)
    private static native void HuMoments_0(double m_m00, double m_m10, double m_m01, double m_m20, double m_m11, double m_m02, double m_m30, double m_m21, double m_m12, double m_m03, long hu_nativeObj);

    // C++:  void cv::Laplacian(Mat src, Mat& dst, int ddepth, int ksize = 1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
    private static native void Laplacian_0(long src_nativeObj, long dst_nativeObj, int ddepth, int ksize, double scale, double delta, int borderType);
    private static native void Laplacian_1(long src_nativeObj, long dst_nativeObj, int ddepth, int ksize, double scale, double delta);
    private static native void Laplacian_2(long src_nativeObj, long dst_nativeObj, int ddepth, int ksize, double scale);
    private static native void Laplacian_3(long src_nativeObj, long dst_nativeObj, int ddepth, int ksize);
    private static native void Laplacian_4(long src_nativeObj, long dst_nativeObj, int ddepth);

    // C++:  void cv::Scharr(Mat src, Mat& dst, int ddepth, int dx, int dy, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
    private static native void Scharr_0(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy, double scale, double delta, int borderType);
    private static native void Scharr_1(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy, double scale, double delta);
    private static native void Scharr_2(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy, double scale);
    private static native void Scharr_3(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy);

    // C++:  void cv::Sobel(Mat src, Mat& dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
    private static native void Sobel_0(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType);
    private static native void Sobel_1(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy, int ksize, double scale, double delta);
    private static native void Sobel_2(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy, int ksize, double scale);
    private static native void Sobel_3(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy, int ksize);
    private static native void Sobel_4(long src_nativeObj, long dst_nativeObj, int ddepth, int dx, int dy);

    // C++:  void cv::accumulate(Mat src, Mat& dst, Mat mask = Mat())
    private static native void accumulate_0(long src_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void accumulate_1(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::accumulateProduct(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
    private static native void accumulateProduct_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void accumulateProduct_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::accumulateSquare(Mat src, Mat& dst, Mat mask = Mat())
    private static native void accumulateSquare_0(long src_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void accumulateSquare_1(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::accumulateWeighted(Mat src, Mat& dst, double alpha, Mat mask = Mat())
    private static native void accumulateWeighted_0(long src_nativeObj, long dst_nativeObj, double alpha, long mask_nativeObj);
    private static native void accumulateWeighted_1(long src_nativeObj, long dst_nativeObj, double alpha);

    // C++:  void cv::adaptiveThreshold(Mat src, Mat& dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C)
    private static native void adaptiveThreshold_0(long src_nativeObj, long dst_nativeObj, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C);

    // C++:  void cv::applyColorMap(Mat src, Mat& dst, Mat userColor)
    private static native void applyColorMap_0(long src_nativeObj, long dst_nativeObj, long userColor_nativeObj);

    // C++:  void cv::applyColorMap(Mat src, Mat& dst, int colormap)
    private static native void applyColorMap_1(long src_nativeObj, long dst_nativeObj, int colormap);

    // C++:  void cv::approxPolyDP(vector_Point2f curve, vector_Point2f& approxCurve, double epsilon, bool closed)
    private static native void approxPolyDP_0(long curve_mat_nativeObj, long approxCurve_mat_nativeObj, double epsilon, boolean closed);

    // C++:  void cv::arrowedLine(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1)
    private static native void arrowedLine_0(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int line_type, int shift, double tipLength);
    private static native void arrowedLine_1(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int line_type, int shift);
    private static native void arrowedLine_2(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int line_type);
    private static native void arrowedLine_3(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void arrowedLine_4(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::bilateralFilter(Mat src, Mat& dst, int d, double sigmaColor, double sigmaSpace, int borderType = BORDER_DEFAULT)
    private static native void bilateralFilter_0(long src_nativeObj, long dst_nativeObj, int d, double sigmaColor, double sigmaSpace, int borderType);
    private static native void bilateralFilter_1(long src_nativeObj, long dst_nativeObj, int d, double sigmaColor, double sigmaSpace);

    // C++:  void cv::blur(Mat src, Mat& dst, Size ksize, Point anchor = Point(-1,-1), int borderType = BORDER_DEFAULT)
    private static native void blur_0(long src_nativeObj, long dst_nativeObj, double ksize_width, double ksize_height, double anchor_x, double anchor_y, int borderType);
    private static native void blur_1(long src_nativeObj, long dst_nativeObj, double ksize_width, double ksize_height, double anchor_x, double anchor_y);
    private static native void blur_2(long src_nativeObj, long dst_nativeObj, double ksize_width, double ksize_height);

    // C++:  void cv::boxFilter(Mat src, Mat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), bool normalize = true, int borderType = BORDER_DEFAULT)
    private static native void boxFilter_0(long src_nativeObj, long dst_nativeObj, int ddepth, double ksize_width, double ksize_height, double anchor_x, double anchor_y, boolean normalize, int borderType);
    private static native void boxFilter_1(long src_nativeObj, long dst_nativeObj, int ddepth, double ksize_width, double ksize_height, double anchor_x, double anchor_y, boolean normalize);
    private static native void boxFilter_2(long src_nativeObj, long dst_nativeObj, int ddepth, double ksize_width, double ksize_height, double anchor_x, double anchor_y);
    private static native void boxFilter_3(long src_nativeObj, long dst_nativeObj, int ddepth, double ksize_width, double ksize_height);

    // C++:  void cv::boxPoints(RotatedRect box, Mat& points)
    private static native void boxPoints_0(double box_center_x, double box_center_y, double box_size_width, double box_size_height, double box_angle, long points_nativeObj);

    // C++:  void cv::calcBackProject(vector_Mat images, vector_int channels, Mat hist, Mat& dst, vector_float ranges, double scale)
    private static native void calcBackProject_0(long images_mat_nativeObj, long channels_mat_nativeObj, long hist_nativeObj, long dst_nativeObj, long ranges_mat_nativeObj, double scale);

    // C++:  void cv::calcHist(vector_Mat images, vector_int channels, Mat mask, Mat& hist, vector_int histSize, vector_float ranges, bool accumulate = false)
    private static native void calcHist_0(long images_mat_nativeObj, long channels_mat_nativeObj, long mask_nativeObj, long hist_nativeObj, long histSize_mat_nativeObj, long ranges_mat_nativeObj, boolean accumulate);
    private static native void calcHist_1(long images_mat_nativeObj, long channels_mat_nativeObj, long mask_nativeObj, long hist_nativeObj, long histSize_mat_nativeObj, long ranges_mat_nativeObj);

    // C++:  void cv::circle(Mat& img, Point center, int radius, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    private static native void circle_0(long img_nativeObj, double center_x, double center_y, int radius, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, int shift);
    private static native void circle_1(long img_nativeObj, double center_x, double center_y, int radius, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void circle_2(long img_nativeObj, double center_x, double center_y, int radius, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void circle_3(long img_nativeObj, double center_x, double center_y, int radius, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::convertMaps(Mat map1, Mat map2, Mat& dstmap1, Mat& dstmap2, int dstmap1type, bool nninterpolation = false)
    private static native void convertMaps_0(long map1_nativeObj, long map2_nativeObj, long dstmap1_nativeObj, long dstmap2_nativeObj, int dstmap1type, boolean nninterpolation);
    private static native void convertMaps_1(long map1_nativeObj, long map2_nativeObj, long dstmap1_nativeObj, long dstmap2_nativeObj, int dstmap1type);

    // C++:  void cv::convexHull(vector_Point points, vector_int& hull, bool clockwise = false,  _hidden_  returnPoints = true)
    private static native void convexHull_0(long points_mat_nativeObj, long hull_mat_nativeObj, boolean clockwise);
    private static native void convexHull_2(long points_mat_nativeObj, long hull_mat_nativeObj);

    // C++:  void cv::convexityDefects(vector_Point contour, vector_int convexhull, vector_Vec4i& convexityDefects)
    private static native void convexityDefects_0(long contour_mat_nativeObj, long convexhull_mat_nativeObj, long convexityDefects_mat_nativeObj);

    // C++:  void cv::cornerEigenValsAndVecs(Mat src, Mat& dst, int blockSize, int ksize, int borderType = BORDER_DEFAULT)
    private static native void cornerEigenValsAndVecs_0(long src_nativeObj, long dst_nativeObj, int blockSize, int ksize, int borderType);
    private static native void cornerEigenValsAndVecs_1(long src_nativeObj, long dst_nativeObj, int blockSize, int ksize);

    // C++:  void cv::cornerHarris(Mat src, Mat& dst, int blockSize, int ksize, double k, int borderType = BORDER_DEFAULT)
    private static native void cornerHarris_0(long src_nativeObj, long dst_nativeObj, int blockSize, int ksize, double k, int borderType);
    private static native void cornerHarris_1(long src_nativeObj, long dst_nativeObj, int blockSize, int ksize, double k);

    // C++:  void cv::cornerMinEigenVal(Mat src, Mat& dst, int blockSize, int ksize = 3, int borderType = BORDER_DEFAULT)
    private static native void cornerMinEigenVal_0(long src_nativeObj, long dst_nativeObj, int blockSize, int ksize, int borderType);
    private static native void cornerMinEigenVal_1(long src_nativeObj, long dst_nativeObj, int blockSize, int ksize);
    private static native void cornerMinEigenVal_2(long src_nativeObj, long dst_nativeObj, int blockSize);

    // C++:  void cv::cornerSubPix(Mat image, Mat& corners, Size winSize, Size zeroZone, TermCriteria criteria)
    private static native void cornerSubPix_0(long image_nativeObj, long corners_nativeObj, double winSize_width, double winSize_height, double zeroZone_width, double zeroZone_height, int criteria_type, int criteria_maxCount, double criteria_epsilon);

    // C++:  void cv::createHanningWindow(Mat& dst, Size winSize, int type)
    private static native void createHanningWindow_0(long dst_nativeObj, double winSize_width, double winSize_height, int type);

    // C++:  void cv::cvtColor(Mat src, Mat& dst, int code, int dstCn = 0)
    private static native void cvtColor_0(long src_nativeObj, long dst_nativeObj, int code, int dstCn);
    private static native void cvtColor_1(long src_nativeObj, long dst_nativeObj, int code);

    // C++:  void cv::cvtColorTwoPlane(Mat src1, Mat src2, Mat& dst, int code)
    private static native void cvtColorTwoPlane_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, int code);

    // C++:  void cv::demosaicing(Mat src, Mat& dst, int code, int dstCn = 0)
    private static native void demosaicing_0(long src_nativeObj, long dst_nativeObj, int code, int dstCn);
    private static native void demosaicing_1(long src_nativeObj, long dst_nativeObj, int code);

    // C++:  void cv::dilate(Mat src, Mat& dst, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
    private static native void dilate_0(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations, int borderType, double borderValue_val0, double borderValue_val1, double borderValue_val2, double borderValue_val3);
    private static native void dilate_1(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations, int borderType);
    private static native void dilate_2(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations);
    private static native void dilate_3(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj, double anchor_x, double anchor_y);
    private static native void dilate_4(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj);

    // C++:  void cv::distanceTransform(Mat src, Mat& dst, Mat& labels, int distanceType, int maskSize, int labelType = DIST_LABEL_CCOMP)
    private static native void distanceTransformWithLabels_0(long src_nativeObj, long dst_nativeObj, long labels_nativeObj, int distanceType, int maskSize, int labelType);
    private static native void distanceTransformWithLabels_1(long src_nativeObj, long dst_nativeObj, long labels_nativeObj, int distanceType, int maskSize);

    // C++:  void cv::distanceTransform(Mat src, Mat& dst, int distanceType, int maskSize, int dstType = CV_32F)
    private static native void distanceTransform_0(long src_nativeObj, long dst_nativeObj, int distanceType, int maskSize, int dstType);
    private static native void distanceTransform_1(long src_nativeObj, long dst_nativeObj, int distanceType, int maskSize);

    // C++:  void cv::drawContours(Mat& image, vector_vector_Point contours, int contourIdx, Scalar color, int thickness = 1, int lineType = LINE_8, Mat hierarchy = Mat(), int maxLevel = INT_MAX, Point offset = Point())
    private static native void drawContours_0(long image_nativeObj, long contours_mat_nativeObj, int contourIdx, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, long hierarchy_nativeObj, int maxLevel, double offset_x, double offset_y);
    private static native void drawContours_1(long image_nativeObj, long contours_mat_nativeObj, int contourIdx, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, long hierarchy_nativeObj, int maxLevel);
    private static native void drawContours_2(long image_nativeObj, long contours_mat_nativeObj, int contourIdx, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, long hierarchy_nativeObj);
    private static native void drawContours_3(long image_nativeObj, long contours_mat_nativeObj, int contourIdx, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void drawContours_4(long image_nativeObj, long contours_mat_nativeObj, int contourIdx, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void drawContours_5(long image_nativeObj, long contours_mat_nativeObj, int contourIdx, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::drawMarker(Mat& img, Point position, Scalar color, int markerType = MARKER_CROSS, int markerSize = 20, int thickness = 1, int line_type = 8)
    private static native void drawMarker_0(long img_nativeObj, double position_x, double position_y, double color_val0, double color_val1, double color_val2, double color_val3, int markerType, int markerSize, int thickness, int line_type);
    private static native void drawMarker_1(long img_nativeObj, double position_x, double position_y, double color_val0, double color_val1, double color_val2, double color_val3, int markerType, int markerSize, int thickness);
    private static native void drawMarker_2(long img_nativeObj, double position_x, double position_y, double color_val0, double color_val1, double color_val2, double color_val3, int markerType, int markerSize);
    private static native void drawMarker_3(long img_nativeObj, double position_x, double position_y, double color_val0, double color_val1, double color_val2, double color_val3, int markerType);
    private static native void drawMarker_4(long img_nativeObj, double position_x, double position_y, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::ellipse(Mat& img, Point center, Size axes, double angle, double startAngle, double endAngle, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    private static native void ellipse_0(long img_nativeObj, double center_x, double center_y, double axes_width, double axes_height, double angle, double startAngle, double endAngle, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, int shift);
    private static native void ellipse_1(long img_nativeObj, double center_x, double center_y, double axes_width, double axes_height, double angle, double startAngle, double endAngle, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void ellipse_2(long img_nativeObj, double center_x, double center_y, double axes_width, double axes_height, double angle, double startAngle, double endAngle, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void ellipse_3(long img_nativeObj, double center_x, double center_y, double axes_width, double axes_height, double angle, double startAngle, double endAngle, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::ellipse(Mat& img, RotatedRect box, Scalar color, int thickness = 1, int lineType = LINE_8)
    private static native void ellipse_4(long img_nativeObj, double box_center_x, double box_center_y, double box_size_width, double box_size_height, double box_angle, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void ellipse_5(long img_nativeObj, double box_center_x, double box_center_y, double box_size_width, double box_size_height, double box_angle, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void ellipse_6(long img_nativeObj, double box_center_x, double box_center_y, double box_size_width, double box_size_height, double box_angle, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::ellipse2Poly(Point center, Size axes, int angle, int arcStart, int arcEnd, int delta, vector_Point& pts)
    private static native void ellipse2Poly_0(double center_x, double center_y, double axes_width, double axes_height, int angle, int arcStart, int arcEnd, int delta, long pts_mat_nativeObj);

    // C++:  void cv::equalizeHist(Mat src, Mat& dst)
    private static native void equalizeHist_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::erode(Mat src, Mat& dst, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
    private static native void erode_0(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations, int borderType, double borderValue_val0, double borderValue_val1, double borderValue_val2, double borderValue_val3);
    private static native void erode_1(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations, int borderType);
    private static native void erode_2(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations);
    private static native void erode_3(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj, double anchor_x, double anchor_y);
    private static native void erode_4(long src_nativeObj, long dst_nativeObj, long kernel_nativeObj);

    // C++:  void cv::fillConvexPoly(Mat& img, vector_Point points, Scalar color, int lineType = LINE_8, int shift = 0)
    private static native void fillConvexPoly_0(long img_nativeObj, long points_mat_nativeObj, double color_val0, double color_val1, double color_val2, double color_val3, int lineType, int shift);
    private static native void fillConvexPoly_1(long img_nativeObj, long points_mat_nativeObj, double color_val0, double color_val1, double color_val2, double color_val3, int lineType);
    private static native void fillConvexPoly_2(long img_nativeObj, long points_mat_nativeObj, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::fillPoly(Mat& img, vector_vector_Point pts, Scalar color, int lineType = LINE_8, int shift = 0, Point offset = Point())
    private static native void fillPoly_0(long img_nativeObj, long pts_mat_nativeObj, double color_val0, double color_val1, double color_val2, double color_val3, int lineType, int shift, double offset_x, double offset_y);
    private static native void fillPoly_1(long img_nativeObj, long pts_mat_nativeObj, double color_val0, double color_val1, double color_val2, double color_val3, int lineType, int shift);
    private static native void fillPoly_2(long img_nativeObj, long pts_mat_nativeObj, double color_val0, double color_val1, double color_val2, double color_val3, int lineType);
    private static native void fillPoly_3(long img_nativeObj, long pts_mat_nativeObj, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::filter2D(Mat src, Mat& dst, int ddepth, Mat kernel, Point anchor = Point(-1,-1), double delta = 0, int borderType = BORDER_DEFAULT)
    private static native void filter2D_0(long src_nativeObj, long dst_nativeObj, int ddepth, long kernel_nativeObj, double anchor_x, double anchor_y, double delta, int borderType);
    private static native void filter2D_1(long src_nativeObj, long dst_nativeObj, int ddepth, long kernel_nativeObj, double anchor_x, double anchor_y, double delta);
    private static native void filter2D_2(long src_nativeObj, long dst_nativeObj, int ddepth, long kernel_nativeObj, double anchor_x, double anchor_y);
    private static native void filter2D_3(long src_nativeObj, long dst_nativeObj, int ddepth, long kernel_nativeObj);

    // C++:  void cv::findContours(Mat image, vector_vector_Point& contours, Mat& hierarchy, int mode, int method, Point offset = Point())
    private static native void findContours_0(long image_nativeObj, long contours_mat_nativeObj, long hierarchy_nativeObj, int mode, int method, double offset_x, double offset_y);
    private static native void findContours_1(long image_nativeObj, long contours_mat_nativeObj, long hierarchy_nativeObj, int mode, int method);

    // C++:  void cv::fitLine(Mat points, Mat& line, int distType, double param, double reps, double aeps)
    private static native void fitLine_0(long points_nativeObj, long line_nativeObj, int distType, double param, double reps, double aeps);

    // C++:  void cv::getDerivKernels(Mat& kx, Mat& ky, int dx, int dy, int ksize, bool normalize = false, int ktype = CV_32F)
    private static native void getDerivKernels_0(long kx_nativeObj, long ky_nativeObj, int dx, int dy, int ksize, boolean normalize, int ktype);
    private static native void getDerivKernels_1(long kx_nativeObj, long ky_nativeObj, int dx, int dy, int ksize, boolean normalize);
    private static native void getDerivKernels_2(long kx_nativeObj, long ky_nativeObj, int dx, int dy, int ksize);

    // C++:  void cv::getRectSubPix(Mat image, Size patchSize, Point2f center, Mat& patch, int patchType = -1)
    private static native void getRectSubPix_0(long image_nativeObj, double patchSize_width, double patchSize_height, double center_x, double center_y, long patch_nativeObj, int patchType);
    private static native void getRectSubPix_1(long image_nativeObj, double patchSize_width, double patchSize_height, double center_x, double center_y, long patch_nativeObj);

    // C++:  void cv::goodFeaturesToTrack(Mat image, vector_Point& corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize, int gradientSize, bool useHarrisDetector = false, double k = 0.04)
    private static native void goodFeaturesToTrack_0(long image_nativeObj, long corners_mat_nativeObj, int maxCorners, double qualityLevel, double minDistance, long mask_nativeObj, int blockSize, int gradientSize, boolean useHarrisDetector, double k);
    private static native void goodFeaturesToTrack_1(long image_nativeObj, long corners_mat_nativeObj, int maxCorners, double qualityLevel, double minDistance, long mask_nativeObj, int blockSize, int gradientSize, boolean useHarrisDetector);
    private static native void goodFeaturesToTrack_2(long image_nativeObj, long corners_mat_nativeObj, int maxCorners, double qualityLevel, double minDistance, long mask_nativeObj, int blockSize, int gradientSize);

    // C++:  void cv::goodFeaturesToTrack(Mat image, vector_Point& corners, int maxCorners, double qualityLevel, double minDistance, Mat mask = Mat(), int blockSize = 3, bool useHarrisDetector = false, double k = 0.04)
    private static native void goodFeaturesToTrack_3(long image_nativeObj, long corners_mat_nativeObj, int maxCorners, double qualityLevel, double minDistance, long mask_nativeObj, int blockSize, boolean useHarrisDetector, double k);
    private static native void goodFeaturesToTrack_4(long image_nativeObj, long corners_mat_nativeObj, int maxCorners, double qualityLevel, double minDistance, long mask_nativeObj, int blockSize, boolean useHarrisDetector);
    private static native void goodFeaturesToTrack_5(long image_nativeObj, long corners_mat_nativeObj, int maxCorners, double qualityLevel, double minDistance, long mask_nativeObj, int blockSize);
    private static native void goodFeaturesToTrack_6(long image_nativeObj, long corners_mat_nativeObj, int maxCorners, double qualityLevel, double minDistance, long mask_nativeObj);
    private static native void goodFeaturesToTrack_7(long image_nativeObj, long corners_mat_nativeObj, int maxCorners, double qualityLevel, double minDistance);

    // C++:  void cv::grabCut(Mat img, Mat& mask, Rect rect, Mat& bgdModel, Mat& fgdModel, int iterCount, int mode = GC_EVAL)
    private static native void grabCut_0(long img_nativeObj, long mask_nativeObj, int rect_x, int rect_y, int rect_width, int rect_height, long bgdModel_nativeObj, long fgdModel_nativeObj, int iterCount, int mode);
    private static native void grabCut_1(long img_nativeObj, long mask_nativeObj, int rect_x, int rect_y, int rect_width, int rect_height, long bgdModel_nativeObj, long fgdModel_nativeObj, int iterCount);

    // C++:  void cv::integral(Mat src, Mat& sum, Mat& sqsum, Mat& tilted, int sdepth = -1, int sqdepth = -1)
    private static native void integral3_0(long src_nativeObj, long sum_nativeObj, long sqsum_nativeObj, long tilted_nativeObj, int sdepth, int sqdepth);
    private static native void integral3_1(long src_nativeObj, long sum_nativeObj, long sqsum_nativeObj, long tilted_nativeObj, int sdepth);
    private static native void integral3_2(long src_nativeObj, long sum_nativeObj, long sqsum_nativeObj, long tilted_nativeObj);

    // C++:  void cv::integral(Mat src, Mat& sum, Mat& sqsum, int sdepth = -1, int sqdepth = -1)
    private static native void integral2_0(long src_nativeObj, long sum_nativeObj, long sqsum_nativeObj, int sdepth, int sqdepth);
    private static native void integral2_1(long src_nativeObj, long sum_nativeObj, long sqsum_nativeObj, int sdepth);
    private static native void integral2_2(long src_nativeObj, long sum_nativeObj, long sqsum_nativeObj);

    // C++:  void cv::integral(Mat src, Mat& sum, int sdepth = -1)
    private static native void integral_0(long src_nativeObj, long sum_nativeObj, int sdepth);
    private static native void integral_1(long src_nativeObj, long sum_nativeObj);

    // C++:  void cv::invertAffineTransform(Mat M, Mat& iM)
    private static native void invertAffineTransform_0(long M_nativeObj, long iM_nativeObj);

    // C++:  void cv::line(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    private static native void line_0(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, int shift);
    private static native void line_1(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void line_2(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void line_3(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::linearPolar(Mat src, Mat& dst, Point2f center, double maxRadius, int flags)
    private static native void linearPolar_0(long src_nativeObj, long dst_nativeObj, double center_x, double center_y, double maxRadius, int flags);

    // C++:  void cv::logPolar(Mat src, Mat& dst, Point2f center, double M, int flags)
    private static native void logPolar_0(long src_nativeObj, long dst_nativeObj, double center_x, double center_y, double M, int flags);

    // C++:  void cv::matchTemplate(Mat image, Mat templ, Mat& result, int method, Mat mask = Mat())
    private static native void matchTemplate_0(long image_nativeObj, long templ_nativeObj, long result_nativeObj, int method, long mask_nativeObj);
    private static native void matchTemplate_1(long image_nativeObj, long templ_nativeObj, long result_nativeObj, int method);

    // C++:  void cv::medianBlur(Mat src, Mat& dst, int ksize)
    private static native void medianBlur_0(long src_nativeObj, long dst_nativeObj, int ksize);

    // C++:  void cv::minEnclosingCircle(vector_Point2f points, Point2f& center, float& radius)
    private static native void minEnclosingCircle_0(long points_mat_nativeObj, double[] center_out, double[] radius_out);

    // C++:  void cv::morphologyEx(Mat src, Mat& dst, int op, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
    private static native void morphologyEx_0(long src_nativeObj, long dst_nativeObj, int op, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations, int borderType, double borderValue_val0, double borderValue_val1, double borderValue_val2, double borderValue_val3);
    private static native void morphologyEx_1(long src_nativeObj, long dst_nativeObj, int op, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations, int borderType);
    private static native void morphologyEx_2(long src_nativeObj, long dst_nativeObj, int op, long kernel_nativeObj, double anchor_x, double anchor_y, int iterations);
    private static native void morphologyEx_3(long src_nativeObj, long dst_nativeObj, int op, long kernel_nativeObj, double anchor_x, double anchor_y);
    private static native void morphologyEx_4(long src_nativeObj, long dst_nativeObj, int op, long kernel_nativeObj);

    // C++:  void cv::polylines(Mat& img, vector_vector_Point pts, bool isClosed, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    private static native void polylines_0(long img_nativeObj, long pts_mat_nativeObj, boolean isClosed, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, int shift);
    private static native void polylines_1(long img_nativeObj, long pts_mat_nativeObj, boolean isClosed, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void polylines_2(long img_nativeObj, long pts_mat_nativeObj, boolean isClosed, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void polylines_3(long img_nativeObj, long pts_mat_nativeObj, boolean isClosed, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::preCornerDetect(Mat src, Mat& dst, int ksize, int borderType = BORDER_DEFAULT)
    private static native void preCornerDetect_0(long src_nativeObj, long dst_nativeObj, int ksize, int borderType);
    private static native void preCornerDetect_1(long src_nativeObj, long dst_nativeObj, int ksize);

    // C++:  void cv::putText(Mat& img, String text, Point org, int fontFace, double fontScale, Scalar color, int thickness = 1, int lineType = LINE_8, bool bottomLeftOrigin = false)
    private static native void putText_0(long img_nativeObj, String text, double org_x, double org_y, int fontFace, double fontScale, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, boolean bottomLeftOrigin);
    private static native void putText_1(long img_nativeObj, String text, double org_x, double org_y, int fontFace, double fontScale, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void putText_2(long img_nativeObj, String text, double org_x, double org_y, int fontFace, double fontScale, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void putText_3(long img_nativeObj, String text, double org_x, double org_y, int fontFace, double fontScale, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::pyrDown(Mat src, Mat& dst, Size dstsize = Size(), int borderType = BORDER_DEFAULT)
    private static native void pyrDown_0(long src_nativeObj, long dst_nativeObj, double dstsize_width, double dstsize_height, int borderType);
    private static native void pyrDown_1(long src_nativeObj, long dst_nativeObj, double dstsize_width, double dstsize_height);
    private static native void pyrDown_2(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::pyrMeanShiftFiltering(Mat src, Mat& dst, double sp, double sr, int maxLevel = 1, TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1))
    private static native void pyrMeanShiftFiltering_0(long src_nativeObj, long dst_nativeObj, double sp, double sr, int maxLevel, int termcrit_type, int termcrit_maxCount, double termcrit_epsilon);
    private static native void pyrMeanShiftFiltering_1(long src_nativeObj, long dst_nativeObj, double sp, double sr, int maxLevel);
    private static native void pyrMeanShiftFiltering_2(long src_nativeObj, long dst_nativeObj, double sp, double sr);

    // C++:  void cv::pyrUp(Mat src, Mat& dst, Size dstsize = Size(), int borderType = BORDER_DEFAULT)
    private static native void pyrUp_0(long src_nativeObj, long dst_nativeObj, double dstsize_width, double dstsize_height, int borderType);
    private static native void pyrUp_1(long src_nativeObj, long dst_nativeObj, double dstsize_width, double dstsize_height);
    private static native void pyrUp_2(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::rectangle(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    private static native void rectangle_0(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, int shift);
    private static native void rectangle_1(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void rectangle_2(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void rectangle_3(long img_nativeObj, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::rectangle(Mat& img, Rect rec, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
    private static native void rectangle_4(long img_nativeObj, int rec_x, int rec_y, int rec_width, int rec_height, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType, int shift);
    private static native void rectangle_5(long img_nativeObj, int rec_x, int rec_y, int rec_width, int rec_height, double color_val0, double color_val1, double color_val2, double color_val3, int thickness, int lineType);
    private static native void rectangle_6(long img_nativeObj, int rec_x, int rec_y, int rec_width, int rec_height, double color_val0, double color_val1, double color_val2, double color_val3, int thickness);
    private static native void rectangle_7(long img_nativeObj, int rec_x, int rec_y, int rec_width, int rec_height, double color_val0, double color_val1, double color_val2, double color_val3);

    // C++:  void cv::remap(Mat src, Mat& dst, Mat map1, Mat map2, int interpolation, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
    private static native void remap_0(long src_nativeObj, long dst_nativeObj, long map1_nativeObj, long map2_nativeObj, int interpolation, int borderMode, double borderValue_val0, double borderValue_val1, double borderValue_val2, double borderValue_val3);
    private static native void remap_1(long src_nativeObj, long dst_nativeObj, long map1_nativeObj, long map2_nativeObj, int interpolation, int borderMode);
    private static native void remap_2(long src_nativeObj, long dst_nativeObj, long map1_nativeObj, long map2_nativeObj, int interpolation);

    // C++:  void cv::resize(Mat src, Mat& dst, Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR)
    private static native void resize_0(long src_nativeObj, long dst_nativeObj, double dsize_width, double dsize_height, double fx, double fy, int interpolation);
    private static native void resize_1(long src_nativeObj, long dst_nativeObj, double dsize_width, double dsize_height, double fx, double fy);
    private static native void resize_2(long src_nativeObj, long dst_nativeObj, double dsize_width, double dsize_height, double fx);
    private static native void resize_3(long src_nativeObj, long dst_nativeObj, double dsize_width, double dsize_height);

    // C++:  void cv::sepFilter2D(Mat src, Mat& dst, int ddepth, Mat kernelX, Mat kernelY, Point anchor = Point(-1,-1), double delta = 0, int borderType = BORDER_DEFAULT)
    private static native void sepFilter2D_0(long src_nativeObj, long dst_nativeObj, int ddepth, long kernelX_nativeObj, long kernelY_nativeObj, double anchor_x, double anchor_y, double delta, int borderType);
    private static native void sepFilter2D_1(long src_nativeObj, long dst_nativeObj, int ddepth, long kernelX_nativeObj, long kernelY_nativeObj, double anchor_x, double anchor_y, double delta);
    private static native void sepFilter2D_2(long src_nativeObj, long dst_nativeObj, int ddepth, long kernelX_nativeObj, long kernelY_nativeObj, double anchor_x, double anchor_y);
    private static native void sepFilter2D_3(long src_nativeObj, long dst_nativeObj, int ddepth, long kernelX_nativeObj, long kernelY_nativeObj);

    // C++:  void cv::spatialGradient(Mat src, Mat& dx, Mat& dy, int ksize = 3, int borderType = BORDER_DEFAULT)
    private static native void spatialGradient_0(long src_nativeObj, long dx_nativeObj, long dy_nativeObj, int ksize, int borderType);
    private static native void spatialGradient_1(long src_nativeObj, long dx_nativeObj, long dy_nativeObj, int ksize);
    private static native void spatialGradient_2(long src_nativeObj, long dx_nativeObj, long dy_nativeObj);

    // C++:  void cv::sqrBoxFilter(Mat src, Mat& dst, int ddepth, Size ksize, Point anchor = Point(-1, -1), bool normalize = true, int borderType = BORDER_DEFAULT)
    private static native void sqrBoxFilter_0(long src_nativeObj, long dst_nativeObj, int ddepth, double ksize_width, double ksize_height, double anchor_x, double anchor_y, boolean normalize, int borderType);
    private static native void sqrBoxFilter_1(long src_nativeObj, long dst_nativeObj, int ddepth, double ksize_width, double ksize_height, double anchor_x, double anchor_y, boolean normalize);
    private static native void sqrBoxFilter_2(long src_nativeObj, long dst_nativeObj, int ddepth, double ksize_width, double ksize_height, double anchor_x, double anchor_y);
    private static native void sqrBoxFilter_3(long src_nativeObj, long dst_nativeObj, int ddepth, double ksize_width, double ksize_height);

    // C++:  void cv::warpAffine(Mat src, Mat& dst, Mat M, Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
    private static native void warpAffine_0(long src_nativeObj, long dst_nativeObj, long M_nativeObj, double dsize_width, double dsize_height, int flags, int borderMode, double borderValue_val0, double borderValue_val1, double borderValue_val2, double borderValue_val3);
    private static native void warpAffine_1(long src_nativeObj, long dst_nativeObj, long M_nativeObj, double dsize_width, double dsize_height, int flags, int borderMode);
    private static native void warpAffine_2(long src_nativeObj, long dst_nativeObj, long M_nativeObj, double dsize_width, double dsize_height, int flags);
    private static native void warpAffine_3(long src_nativeObj, long dst_nativeObj, long M_nativeObj, double dsize_width, double dsize_height);

    // C++:  void cv::warpPerspective(Mat src, Mat& dst, Mat M, Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
    private static native void warpPerspective_0(long src_nativeObj, long dst_nativeObj, long M_nativeObj, double dsize_width, double dsize_height, int flags, int borderMode, double borderValue_val0, double borderValue_val1, double borderValue_val2, double borderValue_val3);
    private static native void warpPerspective_1(long src_nativeObj, long dst_nativeObj, long M_nativeObj, double dsize_width, double dsize_height, int flags, int borderMode);
    private static native void warpPerspective_2(long src_nativeObj, long dst_nativeObj, long M_nativeObj, double dsize_width, double dsize_height, int flags);
    private static native void warpPerspective_3(long src_nativeObj, long dst_nativeObj, long M_nativeObj, double dsize_width, double dsize_height);

    // C++:  void cv::warpPolar(Mat src, Mat& dst, Size dsize, Point2f center, double maxRadius, int flags)
    private static native void warpPolar_0(long src_nativeObj, long dst_nativeObj, double dsize_width, double dsize_height, double center_x, double center_y, double maxRadius, int flags);

    // C++:  void cv::watershed(Mat image, Mat& markers)
    private static native void watershed_0(long image_nativeObj, long markers_nativeObj);
private static native double[] n_getTextSize(String text, int fontFace, double fontScale, int thickness, int[] baseLine);

}
