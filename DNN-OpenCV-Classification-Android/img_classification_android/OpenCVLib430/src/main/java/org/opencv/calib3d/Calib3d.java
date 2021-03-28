//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.calib3d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.utils.Converters;

// C++: class Calib3d

public class Calib3d {

    // C++: enum <unnamed>
    public static final int
            CV_ITERATIVE = 0,
            CV_EPNP = 1,
            CV_P3P = 2,
            CV_DLS = 3,
            CvLevMarq_DONE = 0,
            CvLevMarq_STARTED = 1,
            CvLevMarq_CALC_J = 2,
            CvLevMarq_CHECK_ERR = 3,
            LMEDS = 4,
            RANSAC = 8,
            RHO = 16,
            CALIB_CB_ADAPTIVE_THRESH = 1,
            CALIB_CB_NORMALIZE_IMAGE = 2,
            CALIB_CB_FILTER_QUADS = 4,
            CALIB_CB_FAST_CHECK = 8,
            CALIB_CB_EXHAUSTIVE = 16,
            CALIB_CB_ACCURACY = 32,
            CALIB_CB_LARGER = 64,
            CALIB_CB_MARKER = 128,
            CALIB_CB_SYMMETRIC_GRID = 1,
            CALIB_CB_ASYMMETRIC_GRID = 2,
            CALIB_CB_CLUSTERING = 4,
            CALIB_NINTRINSIC = 18,
            CALIB_USE_INTRINSIC_GUESS = 0x00001,
            CALIB_FIX_ASPECT_RATIO = 0x00002,
            CALIB_FIX_PRINCIPAL_POINT = 0x00004,
            CALIB_ZERO_TANGENT_DIST = 0x00008,
            CALIB_FIX_FOCAL_LENGTH = 0x00010,
            CALIB_FIX_K1 = 0x00020,
            CALIB_FIX_K2 = 0x00040,
            CALIB_FIX_K3 = 0x00080,
            CALIB_FIX_K4 = 0x00800,
            CALIB_FIX_K5 = 0x01000,
            CALIB_FIX_K6 = 0x02000,
            CALIB_RATIONAL_MODEL = 0x04000,
            CALIB_THIN_PRISM_MODEL = 0x08000,
            CALIB_FIX_S1_S2_S3_S4 = 0x10000,
            CALIB_TILTED_MODEL = 0x40000,
            CALIB_FIX_TAUX_TAUY = 0x80000,
            CALIB_USE_QR = 0x100000,
            CALIB_FIX_TANGENT_DIST = 0x200000,
            CALIB_FIX_INTRINSIC = 0x00100,
            CALIB_SAME_FOCAL_LENGTH = 0x00200,
            CALIB_ZERO_DISPARITY = 0x00400,
            CALIB_USE_LU = (1 << 17),
            CALIB_USE_EXTRINSIC_GUESS = (1 << 22),
            FM_7POINT = 1,
            FM_8POINT = 2,
            FM_LMEDS = 4,
            FM_RANSAC = 8,
            fisheye_CALIB_USE_INTRINSIC_GUESS = 1 << 0,
            fisheye_CALIB_RECOMPUTE_EXTRINSIC = 1 << 1,
            fisheye_CALIB_CHECK_COND = 1 << 2,
            fisheye_CALIB_FIX_SKEW = 1 << 3,
            fisheye_CALIB_FIX_K1 = 1 << 4,
            fisheye_CALIB_FIX_K2 = 1 << 5,
            fisheye_CALIB_FIX_K3 = 1 << 6,
            fisheye_CALIB_FIX_K4 = 1 << 7,
            fisheye_CALIB_FIX_INTRINSIC = 1 << 8,
            fisheye_CALIB_FIX_PRINCIPAL_POINT = 1 << 9;


    // C++: enum UndistortTypes
    public static final int
            PROJ_SPHERICAL_ORTHO = 0,
            PROJ_SPHERICAL_EQRECT = 1;


    // C++: enum SolvePnPMethod
    public static final int
            SOLVEPNP_ITERATIVE = 0,
            SOLVEPNP_EPNP = 1,
            SOLVEPNP_P3P = 2,
            SOLVEPNP_DLS = 3,
            SOLVEPNP_UPNP = 4,
            SOLVEPNP_AP3P = 5,
            SOLVEPNP_IPPE = 6,
            SOLVEPNP_IPPE_SQUARE = 7,
            SOLVEPNP_MAX_COUNT = 7+1;


    // C++: enum HandEyeCalibrationMethod
    public static final int
            CALIB_HAND_EYE_TSAI = 0,
            CALIB_HAND_EYE_PARK = 1,
            CALIB_HAND_EYE_HORAUD = 2,
            CALIB_HAND_EYE_ANDREFF = 3,
            CALIB_HAND_EYE_DANIILIDIS = 4;


    // C++: enum GridType
    public static final int
            CirclesGridFinderParameters_SYMMETRIC_GRID = 0,
            CirclesGridFinderParameters_ASYMMETRIC_GRID = 1;


    //
    // C++:  Mat cv::estimateAffine2D(Mat from, Mat to, Mat& inliers = Mat(), int method = RANSAC, double ransacReprojThreshold = 3, size_t maxIters = 2000, double confidence = 0.99, size_t refineIters = 10)
    //

    /**
     * Computes an optimal affine transformation between two 2D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12}\\
     * a_{21} &amp; a_{22}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * \end{bmatrix}
     * \)
     *
     * @param from First input 2D point set containing \((X,Y)\).
     * @param to Second input 2D point set containing \((x,y)\).
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
     * a point as an inlier. Applies only to RANSAC.
     * @param maxIters The maximum number of robust method iterations.
     * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * @param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation matrix \(2 \times 3\) or empty matrix if transformation
     * could not be estimated. The returned matrix has the following form:
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; b_2\\
     * \end{bmatrix}
     * \)
     *
     * The function estimates an optimal 2D affine transformation between two 2D point sets using the
     * selected robust algorithm.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but needs a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffinePartial2D, getAffineTransform
     */
    public static Mat estimateAffine2D(Mat from, Mat to, Mat inliers, int method, double ransacReprojThreshold, long maxIters, double confidence, long refineIters) {
        return new Mat(estimateAffine2D_0(from.nativeObj, to.nativeObj, inliers.nativeObj, method, ransacReprojThreshold, maxIters, confidence, refineIters));
    }

    /**
     * Computes an optimal affine transformation between two 2D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12}\\
     * a_{21} &amp; a_{22}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * \end{bmatrix}
     * \)
     *
     * @param from First input 2D point set containing \((X,Y)\).
     * @param to Second input 2D point set containing \((x,y)\).
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
     * a point as an inlier. Applies only to RANSAC.
     * @param maxIters The maximum number of robust method iterations.
     * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation matrix \(2 \times 3\) or empty matrix if transformation
     * could not be estimated. The returned matrix has the following form:
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; b_2\\
     * \end{bmatrix}
     * \)
     *
     * The function estimates an optimal 2D affine transformation between two 2D point sets using the
     * selected robust algorithm.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but needs a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffinePartial2D, getAffineTransform
     */
    public static Mat estimateAffine2D(Mat from, Mat to, Mat inliers, int method, double ransacReprojThreshold, long maxIters, double confidence) {
        return new Mat(estimateAffine2D_1(from.nativeObj, to.nativeObj, inliers.nativeObj, method, ransacReprojThreshold, maxIters, confidence));
    }

    /**
     * Computes an optimal affine transformation between two 2D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12}\\
     * a_{21} &amp; a_{22}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * \end{bmatrix}
     * \)
     *
     * @param from First input 2D point set containing \((X,Y)\).
     * @param to Second input 2D point set containing \((x,y)\).
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
     * a point as an inlier. Applies only to RANSAC.
     * @param maxIters The maximum number of robust method iterations.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation matrix \(2 \times 3\) or empty matrix if transformation
     * could not be estimated. The returned matrix has the following form:
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; b_2\\
     * \end{bmatrix}
     * \)
     *
     * The function estimates an optimal 2D affine transformation between two 2D point sets using the
     * selected robust algorithm.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but needs a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffinePartial2D, getAffineTransform
     */
    public static Mat estimateAffine2D(Mat from, Mat to, Mat inliers, int method, double ransacReprojThreshold, long maxIters) {
        return new Mat(estimateAffine2D_2(from.nativeObj, to.nativeObj, inliers.nativeObj, method, ransacReprojThreshold, maxIters));
    }

    /**
     * Computes an optimal affine transformation between two 2D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12}\\
     * a_{21} &amp; a_{22}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * \end{bmatrix}
     * \)
     *
     * @param from First input 2D point set containing \((X,Y)\).
     * @param to Second input 2D point set containing \((x,y)\).
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
     * a point as an inlier. Applies only to RANSAC.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation matrix \(2 \times 3\) or empty matrix if transformation
     * could not be estimated. The returned matrix has the following form:
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; b_2\\
     * \end{bmatrix}
     * \)
     *
     * The function estimates an optimal 2D affine transformation between two 2D point sets using the
     * selected robust algorithm.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but needs a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffinePartial2D, getAffineTransform
     */
    public static Mat estimateAffine2D(Mat from, Mat to, Mat inliers, int method, double ransacReprojThreshold) {
        return new Mat(estimateAffine2D_3(from.nativeObj, to.nativeObj, inliers.nativeObj, method, ransacReprojThreshold));
    }

    /**
     * Computes an optimal affine transformation between two 2D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12}\\
     * a_{21} &amp; a_{22}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * \end{bmatrix}
     * \)
     *
     * @param from First input 2D point set containing \((X,Y)\).
     * @param to Second input 2D point set containing \((x,y)\).
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * a point as an inlier. Applies only to RANSAC.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation matrix \(2 \times 3\) or empty matrix if transformation
     * could not be estimated. The returned matrix has the following form:
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; b_2\\
     * \end{bmatrix}
     * \)
     *
     * The function estimates an optimal 2D affine transformation between two 2D point sets using the
     * selected robust algorithm.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but needs a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffinePartial2D, getAffineTransform
     */
    public static Mat estimateAffine2D(Mat from, Mat to, Mat inliers, int method) {
        return new Mat(estimateAffine2D_4(from.nativeObj, to.nativeObj, inliers.nativeObj, method));
    }

    /**
     * Computes an optimal affine transformation between two 2D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12}\\
     * a_{21} &amp; a_{22}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * \end{bmatrix}
     * \)
     *
     * @param from First input 2D point set containing \((X,Y)\).
     * @param to Second input 2D point set containing \((x,y)\).
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * a point as an inlier. Applies only to RANSAC.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation matrix \(2 \times 3\) or empty matrix if transformation
     * could not be estimated. The returned matrix has the following form:
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; b_2\\
     * \end{bmatrix}
     * \)
     *
     * The function estimates an optimal 2D affine transformation between two 2D point sets using the
     * selected robust algorithm.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but needs a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffinePartial2D, getAffineTransform
     */
    public static Mat estimateAffine2D(Mat from, Mat to, Mat inliers) {
        return new Mat(estimateAffine2D_5(from.nativeObj, to.nativeObj, inliers.nativeObj));
    }

    /**
     * Computes an optimal affine transformation between two 2D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12}\\
     * a_{21} &amp; a_{22}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * \end{bmatrix}
     * \)
     *
     * @param from First input 2D point set containing \((X,Y)\).
     * @param to Second input 2D point set containing \((x,y)\).
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * a point as an inlier. Applies only to RANSAC.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation matrix \(2 \times 3\) or empty matrix if transformation
     * could not be estimated. The returned matrix has the following form:
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; b_2\\
     * \end{bmatrix}
     * \)
     *
     * The function estimates an optimal 2D affine transformation between two 2D point sets using the
     * selected robust algorithm.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but needs a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffinePartial2D, getAffineTransform
     */
    public static Mat estimateAffine2D(Mat from, Mat to) {
        return new Mat(estimateAffine2D_6(from.nativeObj, to.nativeObj));
    }


    //
    // C++:  Mat cv::estimateAffinePartial2D(Mat from, Mat to, Mat& inliers = Mat(), int method = RANSAC, double ransacReprojThreshold = 3, size_t maxIters = 2000, double confidence = 0.99, size_t refineIters = 10)
    //

    /**
     * Computes an optimal limited affine transformation with 4 degrees of freedom between
     * two 2D point sets.
     *
     * @param from First input 2D point set.
     * @param to Second input 2D point set.
     * @param inliers Output vector indicating which points are inliers.
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
     * a point as an inlier. Applies only to RANSAC.
     * @param maxIters The maximum number of robust method iterations.
     * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * @param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation (4 degrees of freedom) matrix \(2 \times 3\) or
     * empty matrix if transformation could not be estimated.
     *
     * The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
     * combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
     * estimation.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * Estimated transformation matrix is:
     * \( \begin{bmatrix} \cos(\theta) \cdot s &amp; -\sin(\theta) \cdot s &amp; t_x \\
     *                 \sin(\theta) \cdot s &amp; \cos(\theta) \cdot s &amp; t_y
     * \end{bmatrix} \)
     * Where \( \theta \) is the rotation angle, \( s \) the scaling factor and \( t_x, t_y \) are
     * translations in \( x, y \) axes respectively.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffine2D, getAffineTransform
     */
    public static Mat estimateAffinePartial2D(Mat from, Mat to, Mat inliers, int method, double ransacReprojThreshold, long maxIters, double confidence, long refineIters) {
        return new Mat(estimateAffinePartial2D_0(from.nativeObj, to.nativeObj, inliers.nativeObj, method, ransacReprojThreshold, maxIters, confidence, refineIters));
    }

    /**
     * Computes an optimal limited affine transformation with 4 degrees of freedom between
     * two 2D point sets.
     *
     * @param from First input 2D point set.
     * @param to Second input 2D point set.
     * @param inliers Output vector indicating which points are inliers.
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
     * a point as an inlier. Applies only to RANSAC.
     * @param maxIters The maximum number of robust method iterations.
     * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation (4 degrees of freedom) matrix \(2 \times 3\) or
     * empty matrix if transformation could not be estimated.
     *
     * The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
     * combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
     * estimation.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * Estimated transformation matrix is:
     * \( \begin{bmatrix} \cos(\theta) \cdot s &amp; -\sin(\theta) \cdot s &amp; t_x \\
     *                 \sin(\theta) \cdot s &amp; \cos(\theta) \cdot s &amp; t_y
     * \end{bmatrix} \)
     * Where \( \theta \) is the rotation angle, \( s \) the scaling factor and \( t_x, t_y \) are
     * translations in \( x, y \) axes respectively.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffine2D, getAffineTransform
     */
    public static Mat estimateAffinePartial2D(Mat from, Mat to, Mat inliers, int method, double ransacReprojThreshold, long maxIters, double confidence) {
        return new Mat(estimateAffinePartial2D_1(from.nativeObj, to.nativeObj, inliers.nativeObj, method, ransacReprojThreshold, maxIters, confidence));
    }

    /**
     * Computes an optimal limited affine transformation with 4 degrees of freedom between
     * two 2D point sets.
     *
     * @param from First input 2D point set.
     * @param to Second input 2D point set.
     * @param inliers Output vector indicating which points are inliers.
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
     * a point as an inlier. Applies only to RANSAC.
     * @param maxIters The maximum number of robust method iterations.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation (4 degrees of freedom) matrix \(2 \times 3\) or
     * empty matrix if transformation could not be estimated.
     *
     * The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
     * combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
     * estimation.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * Estimated transformation matrix is:
     * \( \begin{bmatrix} \cos(\theta) \cdot s &amp; -\sin(\theta) \cdot s &amp; t_x \\
     *                 \sin(\theta) \cdot s &amp; \cos(\theta) \cdot s &amp; t_y
     * \end{bmatrix} \)
     * Where \( \theta \) is the rotation angle, \( s \) the scaling factor and \( t_x, t_y \) are
     * translations in \( x, y \) axes respectively.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffine2D, getAffineTransform
     */
    public static Mat estimateAffinePartial2D(Mat from, Mat to, Mat inliers, int method, double ransacReprojThreshold, long maxIters) {
        return new Mat(estimateAffinePartial2D_2(from.nativeObj, to.nativeObj, inliers.nativeObj, method, ransacReprojThreshold, maxIters));
    }

    /**
     * Computes an optimal limited affine transformation with 4 degrees of freedom between
     * two 2D point sets.
     *
     * @param from First input 2D point set.
     * @param to Second input 2D point set.
     * @param inliers Output vector indicating which points are inliers.
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
     * a point as an inlier. Applies only to RANSAC.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation (4 degrees of freedom) matrix \(2 \times 3\) or
     * empty matrix if transformation could not be estimated.
     *
     * The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
     * combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
     * estimation.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * Estimated transformation matrix is:
     * \( \begin{bmatrix} \cos(\theta) \cdot s &amp; -\sin(\theta) \cdot s &amp; t_x \\
     *                 \sin(\theta) \cdot s &amp; \cos(\theta) \cdot s &amp; t_y
     * \end{bmatrix} \)
     * Where \( \theta \) is the rotation angle, \( s \) the scaling factor and \( t_x, t_y \) are
     * translations in \( x, y \) axes respectively.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffine2D, getAffineTransform
     */
    public static Mat estimateAffinePartial2D(Mat from, Mat to, Mat inliers, int method, double ransacReprojThreshold) {
        return new Mat(estimateAffinePartial2D_3(from.nativeObj, to.nativeObj, inliers.nativeObj, method, ransacReprojThreshold));
    }

    /**
     * Computes an optimal limited affine transformation with 4 degrees of freedom between
     * two 2D point sets.
     *
     * @param from First input 2D point set.
     * @param to Second input 2D point set.
     * @param inliers Output vector indicating which points are inliers.
     * @param method Robust method used to compute transformation. The following methods are possible:
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * a point as an inlier. Applies only to RANSAC.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation (4 degrees of freedom) matrix \(2 \times 3\) or
     * empty matrix if transformation could not be estimated.
     *
     * The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
     * combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
     * estimation.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * Estimated transformation matrix is:
     * \( \begin{bmatrix} \cos(\theta) \cdot s &amp; -\sin(\theta) \cdot s &amp; t_x \\
     *                 \sin(\theta) \cdot s &amp; \cos(\theta) \cdot s &amp; t_y
     * \end{bmatrix} \)
     * Where \( \theta \) is the rotation angle, \( s \) the scaling factor and \( t_x, t_y \) are
     * translations in \( x, y \) axes respectively.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffine2D, getAffineTransform
     */
    public static Mat estimateAffinePartial2D(Mat from, Mat to, Mat inliers, int method) {
        return new Mat(estimateAffinePartial2D_4(from.nativeObj, to.nativeObj, inliers.nativeObj, method));
    }

    /**
     * Computes an optimal limited affine transformation with 4 degrees of freedom between
     * two 2D point sets.
     *
     * @param from First input 2D point set.
     * @param to Second input 2D point set.
     * @param inliers Output vector indicating which points are inliers.
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * a point as an inlier. Applies only to RANSAC.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation (4 degrees of freedom) matrix \(2 \times 3\) or
     * empty matrix if transformation could not be estimated.
     *
     * The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
     * combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
     * estimation.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * Estimated transformation matrix is:
     * \( \begin{bmatrix} \cos(\theta) \cdot s &amp; -\sin(\theta) \cdot s &amp; t_x \\
     *                 \sin(\theta) \cdot s &amp; \cos(\theta) \cdot s &amp; t_y
     * \end{bmatrix} \)
     * Where \( \theta \) is the rotation angle, \( s \) the scaling factor and \( t_x, t_y \) are
     * translations in \( x, y \) axes respectively.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffine2D, getAffineTransform
     */
    public static Mat estimateAffinePartial2D(Mat from, Mat to, Mat inliers) {
        return new Mat(estimateAffinePartial2D_5(from.nativeObj, to.nativeObj, inliers.nativeObj));
    }

    /**
     * Computes an optimal limited affine transformation with 4 degrees of freedom between
     * two 2D point sets.
     *
     * @param from First input 2D point set.
     * @param to Second input 2D point set.
     * <ul>
     *   <li>
     *    cv::RANSAC - RANSAC-based robust method
     *   </li>
     *   <li>
     *    cv::LMEDS - Least-Median robust method
     * RANSAC is the default method.
     * a point as an inlier. Applies only to RANSAC.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     * Passing 0 will disable refining, so the output matrix will be output of robust method.
     *   </li>
     * </ul>
     *
     * @return Output 2D affine transformation (4 degrees of freedom) matrix \(2 \times 3\) or
     * empty matrix if transformation could not be estimated.
     *
     * The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
     * combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
     * estimation.
     *
     * The computed transformation is then refined further (using only inliers) with the
     * Levenberg-Marquardt method to reduce the re-projection error even more.
     *
     * Estimated transformation matrix is:
     * \( \begin{bmatrix} \cos(\theta) \cdot s &amp; -\sin(\theta) \cdot s &amp; t_x \\
     *                 \sin(\theta) \cdot s &amp; \cos(\theta) \cdot s &amp; t_y
     * \end{bmatrix} \)
     * Where \( \theta \) is the rotation angle, \( s \) the scaling factor and \( t_x, t_y \) are
     * translations in \( x, y \) axes respectively.
     *
     * <b>Note:</b>
     * The RANSAC method can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers.
     *
     * SEE: estimateAffine2D, getAffineTransform
     */
    public static Mat estimateAffinePartial2D(Mat from, Mat to) {
        return new Mat(estimateAffinePartial2D_6(from.nativeObj, to.nativeObj));
    }


    //
    // C++:  Mat cv::findEssentialMat(Mat points1, Mat points2, Mat cameraMatrix, int method = RANSAC, double prob = 0.999, double threshold = 1.0, Mat& mask = Mat())
    //

    /**
     * Calculates an essential matrix from the corresponding points in two images.
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param cameraMatrix Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param method Method for computing an essential matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * @param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
     * confidence (probability) that the estimated matrix is correct.
     * @param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * @param mask Output array of N elements, every element of which is set to 0 for outliers and to 1
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function estimates essential matrix based on the five-point algorithm solver in CITE: Nister03 .
     * CITE: SteweniusCFS is also a related. The epipolar geometry is described by the following equation:
     *
     * \([p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\)
     *
     * where \(E\) is an essential matrix, \(p_1\) and \(p_2\) are corresponding points in the first and the
     * second images, respectively. The result of this function may be passed further to
     * decomposeEssentialMat or recoverPose to recover the relative pose between cameras.
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, Mat cameraMatrix, int method, double prob, double threshold, Mat mask) {
        return new Mat(findEssentialMat_0(points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, method, prob, threshold, mask.nativeObj));
    }

    /**
     * Calculates an essential matrix from the corresponding points in two images.
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param cameraMatrix Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param method Method for computing an essential matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * @param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
     * confidence (probability) that the estimated matrix is correct.
     * @param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function estimates essential matrix based on the five-point algorithm solver in CITE: Nister03 .
     * CITE: SteweniusCFS is also a related. The epipolar geometry is described by the following equation:
     *
     * \([p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\)
     *
     * where \(E\) is an essential matrix, \(p_1\) and \(p_2\) are corresponding points in the first and the
     * second images, respectively. The result of this function may be passed further to
     * decomposeEssentialMat or recoverPose to recover the relative pose between cameras.
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, Mat cameraMatrix, int method, double prob, double threshold) {
        return new Mat(findEssentialMat_1(points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, method, prob, threshold));
    }

    /**
     * Calculates an essential matrix from the corresponding points in two images.
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param cameraMatrix Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param method Method for computing an essential matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * @param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
     * confidence (probability) that the estimated matrix is correct.
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function estimates essential matrix based on the five-point algorithm solver in CITE: Nister03 .
     * CITE: SteweniusCFS is also a related. The epipolar geometry is described by the following equation:
     *
     * \([p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\)
     *
     * where \(E\) is an essential matrix, \(p_1\) and \(p_2\) are corresponding points in the first and the
     * second images, respectively. The result of this function may be passed further to
     * decomposeEssentialMat or recoverPose to recover the relative pose between cameras.
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, Mat cameraMatrix, int method, double prob) {
        return new Mat(findEssentialMat_2(points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, method, prob));
    }

    /**
     * Calculates an essential matrix from the corresponding points in two images.
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param cameraMatrix Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param method Method for computing an essential matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * confidence (probability) that the estimated matrix is correct.
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function estimates essential matrix based on the five-point algorithm solver in CITE: Nister03 .
     * CITE: SteweniusCFS is also a related. The epipolar geometry is described by the following equation:
     *
     * \([p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\)
     *
     * where \(E\) is an essential matrix, \(p_1\) and \(p_2\) are corresponding points in the first and the
     * second images, respectively. The result of this function may be passed further to
     * decomposeEssentialMat or recoverPose to recover the relative pose between cameras.
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, Mat cameraMatrix, int method) {
        return new Mat(findEssentialMat_3(points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, method));
    }

    /**
     * Calculates an essential matrix from the corresponding points in two images.
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param cameraMatrix Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * confidence (probability) that the estimated matrix is correct.
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function estimates essential matrix based on the five-point algorithm solver in CITE: Nister03 .
     * CITE: SteweniusCFS is also a related. The epipolar geometry is described by the following equation:
     *
     * \([p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\)
     *
     * where \(E\) is an essential matrix, \(p_1\) and \(p_2\) are corresponding points in the first and the
     * second images, respectively. The result of this function may be passed further to
     * decomposeEssentialMat or recoverPose to recover the relative pose between cameras.
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, Mat cameraMatrix) {
        return new Mat(findEssentialMat_4(points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj));
    }


    //
    // C++:  Mat cv::findEssentialMat(Mat points1, Mat points2, double focal = 1.0, Point2d pp = Point2d(0, 0), int method = RANSAC, double prob = 0.999, double threshold = 1.0, Mat& mask = Mat())
    //

    /**
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param focal focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * @param pp principal point of the camera.
     * @param method Method for computing a fundamental matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * @param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * @param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
     * confidence (probability) that the estimated matrix is correct.
     * @param mask Output array of N elements, every element of which is set to 0 for outliers and to 1
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(K =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, double focal, Point pp, int method, double prob, double threshold, Mat mask) {
        return new Mat(findEssentialMat_5(points1.nativeObj, points2.nativeObj, focal, pp.x, pp.y, method, prob, threshold, mask.nativeObj));
    }

    /**
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param focal focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * @param pp principal point of the camera.
     * @param method Method for computing a fundamental matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * @param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * @param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
     * confidence (probability) that the estimated matrix is correct.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(K =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, double focal, Point pp, int method, double prob, double threshold) {
        return new Mat(findEssentialMat_6(points1.nativeObj, points2.nativeObj, focal, pp.x, pp.y, method, prob, threshold));
    }

    /**
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param focal focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * @param pp principal point of the camera.
     * @param method Method for computing a fundamental matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * @param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
     * confidence (probability) that the estimated matrix is correct.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(K =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, double focal, Point pp, int method, double prob) {
        return new Mat(findEssentialMat_7(points1.nativeObj, points2.nativeObj, focal, pp.x, pp.y, method, prob));
    }

    /**
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param focal focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * @param pp principal point of the camera.
     * @param method Method for computing a fundamental matrix.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * confidence (probability) that the estimated matrix is correct.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(K =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, double focal, Point pp, int method) {
        return new Mat(findEssentialMat_8(points1.nativeObj, points2.nativeObj, focal, pp.x, pp.y, method));
    }

    /**
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param focal focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * @param pp principal point of the camera.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * confidence (probability) that the estimated matrix is correct.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(K =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, double focal, Point pp) {
        return new Mat(findEssentialMat_9(points1.nativeObj, points2.nativeObj, focal, pp.x, pp.y));
    }

    /**
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param focal focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * confidence (probability) that the estimated matrix is correct.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(K =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2, double focal) {
        return new Mat(findEssentialMat_10(points1.nativeObj, points2.nativeObj, focal));
    }

    /**
     *
     * @param points1 Array of N (N &gt;= 5) 2D points from the first image. The point coordinates should
     * be floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * are feature points from cameras with same focal length and principal point.
     * <ul>
     *   <li>
     *    <b>RANSAC</b> for the RANSAC algorithm.
     *   </li>
     *   <li>
     *    <b>LMEDS</b> for the LMedS algorithm.
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * confidence (probability) that the estimated matrix is correct.
     * for the other points. The array is computed only in the RANSAC and LMedS methods.
     *   </li>
     * </ul>
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(K =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static Mat findEssentialMat(Mat points1, Mat points2) {
        return new Mat(findEssentialMat_11(points1.nativeObj, points2.nativeObj));
    }


    //
    // C++:  Mat cv::findFundamentalMat(vector_Point2f points1, vector_Point2f points2, int method, double ransacReprojThreshold, double confidence, int maxIters, Mat& mask = Mat())
    //

    /**
     * Calculates a fundamental matrix from the corresponding points in two images.
     *
     * @param points1 Array of N points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param method Method for computing a fundamental matrix.
     * <ul>
     *   <li>
     *    <b>CV_FM_7POINT</b> for a 7-point algorithm. \(N = 7\)
     *   </li>
     *   <li>
     *    <b>CV_FM_8POINT</b> for an 8-point algorithm. \(N \ge 8\)
     *   </li>
     *   <li>
     *    <b>CV_FM_RANSAC</b> for the RANSAC algorithm. \(N \ge 8\)
     *   </li>
     *   <li>
     *    <b>CV_FM_LMEDS</b> for the LMedS algorithm. \(N \ge 8\)
     * @param ransacReprojThreshold Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * @param confidence Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level
     * of confidence (probability) that the estimated matrix is correct.
     * @param mask
     * @param maxIters The maximum number of robust method iterations.
     *   </li>
     * </ul>
     *
     * The epipolar geometry is described by the following equation:
     *
     * \([p_2; 1]^T F [p_1; 1] = 0\)
     *
     * where \(F\) is a fundamental matrix, \(p_1\) and \(p_2\) are corresponding points in the first and the
     * second images, respectively.
     *
     * The function calculates the fundamental matrix using one of four methods listed above and returns
     * the found fundamental matrix. Normally just one matrix is found. But in case of the 7-point
     * algorithm, the function may return up to 3 solutions ( \(9 \times 3\) matrix that stores all 3
     * matrices sequentially).
     *
     * The calculated fundamental matrix may be passed further to computeCorrespondEpilines that finds the
     * epipolar lines corresponding to the specified points. It can also be passed to
     * stereoRectifyUncalibrated to compute the rectification transformation. :
     * <code>
     *     // Example. Estimation of fundamental matrix using the RANSAC algorithm
     *     int point_count = 100;
     *     vector&lt;Point2f&gt; points1(point_count);
     *     vector&lt;Point2f&gt; points2(point_count);
     *
     *     // initialize the points here ...
     *     for( int i = 0; i &lt; point_count; i++ )
     *     {
     *         points1[i] = ...;
     *         points2[i] = ...;
     *     }
     *
     *     Mat fundamental_matrix =
     *      findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
     * </code>
     * @return automatically generated
     */
    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method, double ransacReprojThreshold, double confidence, int maxIters, Mat mask) {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        return new Mat(findFundamentalMat_0(points1_mat.nativeObj, points2_mat.nativeObj, method, ransacReprojThreshold, confidence, maxIters, mask.nativeObj));
    }

    /**
     * Calculates a fundamental matrix from the corresponding points in two images.
     *
     * @param points1 Array of N points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param method Method for computing a fundamental matrix.
     * <ul>
     *   <li>
     *    <b>CV_FM_7POINT</b> for a 7-point algorithm. \(N = 7\)
     *   </li>
     *   <li>
     *    <b>CV_FM_8POINT</b> for an 8-point algorithm. \(N \ge 8\)
     *   </li>
     *   <li>
     *    <b>CV_FM_RANSAC</b> for the RANSAC algorithm. \(N \ge 8\)
     *   </li>
     *   <li>
     *    <b>CV_FM_LMEDS</b> for the LMedS algorithm. \(N \ge 8\)
     * @param ransacReprojThreshold Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar
     * line in pixels, beyond which the point is considered an outlier and is not used for computing the
     * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
     * point localization, image resolution, and the image noise.
     * @param confidence Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level
     * of confidence (probability) that the estimated matrix is correct.
     * @param maxIters The maximum number of robust method iterations.
     *   </li>
     * </ul>
     *
     * The epipolar geometry is described by the following equation:
     *
     * \([p_2; 1]^T F [p_1; 1] = 0\)
     *
     * where \(F\) is a fundamental matrix, \(p_1\) and \(p_2\) are corresponding points in the first and the
     * second images, respectively.
     *
     * The function calculates the fundamental matrix using one of four methods listed above and returns
     * the found fundamental matrix. Normally just one matrix is found. But in case of the 7-point
     * algorithm, the function may return up to 3 solutions ( \(9 \times 3\) matrix that stores all 3
     * matrices sequentially).
     *
     * The calculated fundamental matrix may be passed further to computeCorrespondEpilines that finds the
     * epipolar lines corresponding to the specified points. It can also be passed to
     * stereoRectifyUncalibrated to compute the rectification transformation. :
     * <code>
     *     // Example. Estimation of fundamental matrix using the RANSAC algorithm
     *     int point_count = 100;
     *     vector&lt;Point2f&gt; points1(point_count);
     *     vector&lt;Point2f&gt; points2(point_count);
     *
     *     // initialize the points here ...
     *     for( int i = 0; i &lt; point_count; i++ )
     *     {
     *         points1[i] = ...;
     *         points2[i] = ...;
     *     }
     *
     *     Mat fundamental_matrix =
     *      findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
     * </code>
     * @return automatically generated
     */
    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method, double ransacReprojThreshold, double confidence, int maxIters) {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        return new Mat(findFundamentalMat_1(points1_mat.nativeObj, points2_mat.nativeObj, method, ransacReprojThreshold, confidence, maxIters));
    }


    //
    // C++:  Mat cv::findFundamentalMat(vector_Point2f points1, vector_Point2f points2, int method = FM_RANSAC, double ransacReprojThreshold = 3., double confidence = 0.99, Mat& mask = Mat())
    //

    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method, double ransacReprojThreshold, double confidence, Mat mask) {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        return new Mat(findFundamentalMat_2(points1_mat.nativeObj, points2_mat.nativeObj, method, ransacReprojThreshold, confidence, mask.nativeObj));
    }

    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method, double ransacReprojThreshold, double confidence) {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        return new Mat(findFundamentalMat_3(points1_mat.nativeObj, points2_mat.nativeObj, method, ransacReprojThreshold, confidence));
    }

    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method, double ransacReprojThreshold) {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        return new Mat(findFundamentalMat_4(points1_mat.nativeObj, points2_mat.nativeObj, method, ransacReprojThreshold));
    }

    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method) {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        return new Mat(findFundamentalMat_5(points1_mat.nativeObj, points2_mat.nativeObj, method));
    }

    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2) {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        return new Mat(findFundamentalMat_6(points1_mat.nativeObj, points2_mat.nativeObj));
    }


    //
    // C++:  Mat cv::findHomography(vector_Point2f srcPoints, vector_Point2f dstPoints, int method = 0, double ransacReprojThreshold = 3, Mat& mask = Mat(), int maxIters = 2000, double confidence = 0.995)
    //

    /**
     * Finds a perspective transformation between two planes.
     *
     * @param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
     * or vector&lt;Point2f&gt; .
     * @param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
     * a vector&lt;Point2f&gt; .
     * @param method Method used to compute a homography matrix. The following methods are possible:
     * <ul>
     *   <li>
     *    <b>0</b> - a regular method using all the points, i.e., the least squares method
     *   </li>
     *   <li>
     *    <b>RANSAC</b> - RANSAC-based robust method
     *   </li>
     *   <li>
     *    <b>LMEDS</b> - Least-Median robust method
     *   </li>
     *   <li>
     *    <b>RHO</b> - PROSAC-based robust method
     * @param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
     * (used in the RANSAC and RHO methods only). That is, if
     * \(\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  &gt;  \texttt{ransacReprojThreshold}\)
     * then the point \(i\) is considered as an outlier. If srcPoints and dstPoints are measured in pixels,
     * it usually makes sense to set this parameter somewhere in the range of 1 to 10.
     * @param mask Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input
     * mask values are ignored.
     * @param maxIters The maximum number of RANSAC iterations.
     * @param confidence Confidence level, between 0 and 1.
     *   </li>
     * </ul>
     *
     * The function finds and returns the perspective transformation \(H\) between the source and the
     * destination planes:
     *
     * \(s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\)
     *
     * so that the back-projection error
     *
     * \(\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\)
     *
     * is minimized. If the parameter method is set to the default value 0, the function uses all the point
     * pairs to compute an initial homography estimate with a simple least-squares scheme.
     *
     * However, if not all of the point pairs ( \(srcPoints_i\), \(dstPoints_i\) ) fit the rigid perspective
     * transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
     * you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
     * random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix
     * using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the
     * computed homography (which is the number of inliers for RANSAC or the least median re-projection error for
     * LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and
     * the mask of inliers/outliers.
     *
     * Regardless of the method, robust or not, the computed homography matrix is refined further (using
     * inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
     * re-projection error even more.
     *
     * The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
     * noise is rather small, use the default method (method=0).
     *
     * The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
     * determined up to a scale. Thus, it is normalized so that \(h_{33}=1\). Note that whenever an \(H\) matrix
     * cannot be estimated, an empty one will be returned.
     *
     * SEE:
     * getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
     * perspectiveTransform
     * @return automatically generated
     */
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints, int method, double ransacReprojThreshold, Mat mask, int maxIters, double confidence) {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        return new Mat(findHomography_0(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj, method, ransacReprojThreshold, mask.nativeObj, maxIters, confidence));
    }

    /**
     * Finds a perspective transformation between two planes.
     *
     * @param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
     * or vector&lt;Point2f&gt; .
     * @param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
     * a vector&lt;Point2f&gt; .
     * @param method Method used to compute a homography matrix. The following methods are possible:
     * <ul>
     *   <li>
     *    <b>0</b> - a regular method using all the points, i.e., the least squares method
     *   </li>
     *   <li>
     *    <b>RANSAC</b> - RANSAC-based robust method
     *   </li>
     *   <li>
     *    <b>LMEDS</b> - Least-Median robust method
     *   </li>
     *   <li>
     *    <b>RHO</b> - PROSAC-based robust method
     * @param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
     * (used in the RANSAC and RHO methods only). That is, if
     * \(\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  &gt;  \texttt{ransacReprojThreshold}\)
     * then the point \(i\) is considered as an outlier. If srcPoints and dstPoints are measured in pixels,
     * it usually makes sense to set this parameter somewhere in the range of 1 to 10.
     * @param mask Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input
     * mask values are ignored.
     * @param maxIters The maximum number of RANSAC iterations.
     *   </li>
     * </ul>
     *
     * The function finds and returns the perspective transformation \(H\) between the source and the
     * destination planes:
     *
     * \(s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\)
     *
     * so that the back-projection error
     *
     * \(\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\)
     *
     * is minimized. If the parameter method is set to the default value 0, the function uses all the point
     * pairs to compute an initial homography estimate with a simple least-squares scheme.
     *
     * However, if not all of the point pairs ( \(srcPoints_i\), \(dstPoints_i\) ) fit the rigid perspective
     * transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
     * you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
     * random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix
     * using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the
     * computed homography (which is the number of inliers for RANSAC or the least median re-projection error for
     * LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and
     * the mask of inliers/outliers.
     *
     * Regardless of the method, robust or not, the computed homography matrix is refined further (using
     * inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
     * re-projection error even more.
     *
     * The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
     * noise is rather small, use the default method (method=0).
     *
     * The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
     * determined up to a scale. Thus, it is normalized so that \(h_{33}=1\). Note that whenever an \(H\) matrix
     * cannot be estimated, an empty one will be returned.
     *
     * SEE:
     * getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
     * perspectiveTransform
     * @return automatically generated
     */
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints, int method, double ransacReprojThreshold, Mat mask, int maxIters) {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        return new Mat(findHomography_1(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj, method, ransacReprojThreshold, mask.nativeObj, maxIters));
    }

    /**
     * Finds a perspective transformation between two planes.
     *
     * @param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
     * or vector&lt;Point2f&gt; .
     * @param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
     * a vector&lt;Point2f&gt; .
     * @param method Method used to compute a homography matrix. The following methods are possible:
     * <ul>
     *   <li>
     *    <b>0</b> - a regular method using all the points, i.e., the least squares method
     *   </li>
     *   <li>
     *    <b>RANSAC</b> - RANSAC-based robust method
     *   </li>
     *   <li>
     *    <b>LMEDS</b> - Least-Median robust method
     *   </li>
     *   <li>
     *    <b>RHO</b> - PROSAC-based robust method
     * @param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
     * (used in the RANSAC and RHO methods only). That is, if
     * \(\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  &gt;  \texttt{ransacReprojThreshold}\)
     * then the point \(i\) is considered as an outlier. If srcPoints and dstPoints are measured in pixels,
     * it usually makes sense to set this parameter somewhere in the range of 1 to 10.
     * @param mask Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input
     * mask values are ignored.
     *   </li>
     * </ul>
     *
     * The function finds and returns the perspective transformation \(H\) between the source and the
     * destination planes:
     *
     * \(s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\)
     *
     * so that the back-projection error
     *
     * \(\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\)
     *
     * is minimized. If the parameter method is set to the default value 0, the function uses all the point
     * pairs to compute an initial homography estimate with a simple least-squares scheme.
     *
     * However, if not all of the point pairs ( \(srcPoints_i\), \(dstPoints_i\) ) fit the rigid perspective
     * transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
     * you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
     * random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix
     * using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the
     * computed homography (which is the number of inliers for RANSAC or the least median re-projection error for
     * LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and
     * the mask of inliers/outliers.
     *
     * Regardless of the method, robust or not, the computed homography matrix is refined further (using
     * inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
     * re-projection error even more.
     *
     * The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
     * noise is rather small, use the default method (method=0).
     *
     * The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
     * determined up to a scale. Thus, it is normalized so that \(h_{33}=1\). Note that whenever an \(H\) matrix
     * cannot be estimated, an empty one will be returned.
     *
     * SEE:
     * getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
     * perspectiveTransform
     * @return automatically generated
     */
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints, int method, double ransacReprojThreshold, Mat mask) {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        return new Mat(findHomography_2(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj, method, ransacReprojThreshold, mask.nativeObj));
    }

    /**
     * Finds a perspective transformation between two planes.
     *
     * @param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
     * or vector&lt;Point2f&gt; .
     * @param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
     * a vector&lt;Point2f&gt; .
     * @param method Method used to compute a homography matrix. The following methods are possible:
     * <ul>
     *   <li>
     *    <b>0</b> - a regular method using all the points, i.e., the least squares method
     *   </li>
     *   <li>
     *    <b>RANSAC</b> - RANSAC-based robust method
     *   </li>
     *   <li>
     *    <b>LMEDS</b> - Least-Median robust method
     *   </li>
     *   <li>
     *    <b>RHO</b> - PROSAC-based robust method
     * @param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
     * (used in the RANSAC and RHO methods only). That is, if
     * \(\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  &gt;  \texttt{ransacReprojThreshold}\)
     * then the point \(i\) is considered as an outlier. If srcPoints and dstPoints are measured in pixels,
     * it usually makes sense to set this parameter somewhere in the range of 1 to 10.
     * mask values are ignored.
     *   </li>
     * </ul>
     *
     * The function finds and returns the perspective transformation \(H\) between the source and the
     * destination planes:
     *
     * \(s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\)
     *
     * so that the back-projection error
     *
     * \(\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\)
     *
     * is minimized. If the parameter method is set to the default value 0, the function uses all the point
     * pairs to compute an initial homography estimate with a simple least-squares scheme.
     *
     * However, if not all of the point pairs ( \(srcPoints_i\), \(dstPoints_i\) ) fit the rigid perspective
     * transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
     * you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
     * random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix
     * using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the
     * computed homography (which is the number of inliers for RANSAC or the least median re-projection error for
     * LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and
     * the mask of inliers/outliers.
     *
     * Regardless of the method, robust or not, the computed homography matrix is refined further (using
     * inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
     * re-projection error even more.
     *
     * The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
     * noise is rather small, use the default method (method=0).
     *
     * The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
     * determined up to a scale. Thus, it is normalized so that \(h_{33}=1\). Note that whenever an \(H\) matrix
     * cannot be estimated, an empty one will be returned.
     *
     * SEE:
     * getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
     * perspectiveTransform
     * @return automatically generated
     */
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints, int method, double ransacReprojThreshold) {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        return new Mat(findHomography_3(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj, method, ransacReprojThreshold));
    }

    /**
     * Finds a perspective transformation between two planes.
     *
     * @param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
     * or vector&lt;Point2f&gt; .
     * @param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
     * a vector&lt;Point2f&gt; .
     * @param method Method used to compute a homography matrix. The following methods are possible:
     * <ul>
     *   <li>
     *    <b>0</b> - a regular method using all the points, i.e., the least squares method
     *   </li>
     *   <li>
     *    <b>RANSAC</b> - RANSAC-based robust method
     *   </li>
     *   <li>
     *    <b>LMEDS</b> - Least-Median robust method
     *   </li>
     *   <li>
     *    <b>RHO</b> - PROSAC-based robust method
     * (used in the RANSAC and RHO methods only). That is, if
     * \(\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  &gt;  \texttt{ransacReprojThreshold}\)
     * then the point \(i\) is considered as an outlier. If srcPoints and dstPoints are measured in pixels,
     * it usually makes sense to set this parameter somewhere in the range of 1 to 10.
     * mask values are ignored.
     *   </li>
     * </ul>
     *
     * The function finds and returns the perspective transformation \(H\) between the source and the
     * destination planes:
     *
     * \(s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\)
     *
     * so that the back-projection error
     *
     * \(\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\)
     *
     * is minimized. If the parameter method is set to the default value 0, the function uses all the point
     * pairs to compute an initial homography estimate with a simple least-squares scheme.
     *
     * However, if not all of the point pairs ( \(srcPoints_i\), \(dstPoints_i\) ) fit the rigid perspective
     * transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
     * you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
     * random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix
     * using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the
     * computed homography (which is the number of inliers for RANSAC or the least median re-projection error for
     * LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and
     * the mask of inliers/outliers.
     *
     * Regardless of the method, robust or not, the computed homography matrix is refined further (using
     * inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
     * re-projection error even more.
     *
     * The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
     * noise is rather small, use the default method (method=0).
     *
     * The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
     * determined up to a scale. Thus, it is normalized so that \(h_{33}=1\). Note that whenever an \(H\) matrix
     * cannot be estimated, an empty one will be returned.
     *
     * SEE:
     * getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
     * perspectiveTransform
     * @return automatically generated
     */
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints, int method) {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        return new Mat(findHomography_4(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj, method));
    }

    /**
     * Finds a perspective transformation between two planes.
     *
     * @param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
     * or vector&lt;Point2f&gt; .
     * @param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
     * a vector&lt;Point2f&gt; .
     * <ul>
     *   <li>
     *    <b>0</b> - a regular method using all the points, i.e., the least squares method
     *   </li>
     *   <li>
     *    <b>RANSAC</b> - RANSAC-based robust method
     *   </li>
     *   <li>
     *    <b>LMEDS</b> - Least-Median robust method
     *   </li>
     *   <li>
     *    <b>RHO</b> - PROSAC-based robust method
     * (used in the RANSAC and RHO methods only). That is, if
     * \(\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  &gt;  \texttt{ransacReprojThreshold}\)
     * then the point \(i\) is considered as an outlier. If srcPoints and dstPoints are measured in pixels,
     * it usually makes sense to set this parameter somewhere in the range of 1 to 10.
     * mask values are ignored.
     *   </li>
     * </ul>
     *
     * The function finds and returns the perspective transformation \(H\) between the source and the
     * destination planes:
     *
     * \(s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\)
     *
     * so that the back-projection error
     *
     * \(\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\)
     *
     * is minimized. If the parameter method is set to the default value 0, the function uses all the point
     * pairs to compute an initial homography estimate with a simple least-squares scheme.
     *
     * However, if not all of the point pairs ( \(srcPoints_i\), \(dstPoints_i\) ) fit the rigid perspective
     * transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
     * you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
     * random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix
     * using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the
     * computed homography (which is the number of inliers for RANSAC or the least median re-projection error for
     * LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and
     * the mask of inliers/outliers.
     *
     * Regardless of the method, robust or not, the computed homography matrix is refined further (using
     * inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
     * re-projection error even more.
     *
     * The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
     * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
     * correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
     * noise is rather small, use the default method (method=0).
     *
     * The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
     * determined up to a scale. Thus, it is normalized so that \(h_{33}=1\). Note that whenever an \(H\) matrix
     * cannot be estimated, an empty one will be returned.
     *
     * SEE:
     * getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
     * perspectiveTransform
     * @return automatically generated
     */
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints) {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        return new Mat(findHomography_5(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj));
    }


    //
    // C++:  Mat cv::getDefaultNewCameraMatrix(Mat cameraMatrix, Size imgsize = Size(), bool centerPrincipalPoint = false)
    //

    /**
     * Returns the default new camera matrix.
     *
     * The function returns the camera matrix that is either an exact copy of the input cameraMatrix (when
     * centerPrinicipalPoint=false ), or the modified one (when centerPrincipalPoint=true).
     *
     * In the latter case, the new camera matrix will be:
     *
     * \(\begin{bmatrix} f_x &amp;&amp; 0 &amp;&amp; ( \texttt{imgSize.width} -1)*0.5  \\ 0 &amp;&amp; f_y &amp;&amp; ( \texttt{imgSize.height} -1)*0.5  \\ 0 &amp;&amp; 0 &amp;&amp; 1 \end{bmatrix} ,\)
     *
     * where \(f_x\) and \(f_y\) are \((0,0)\) and \((1,1)\) elements of cameraMatrix, respectively.
     *
     * By default, the undistortion functions in OpenCV (see #initUndistortRectifyMap, #undistort) do not
     * move the principal point. However, when you work with stereo, it is important to move the principal
     * points in both views to the same y-coordinate (which is required by most of stereo correspondence
     * algorithms), and may be to the same x-coordinate too. So, you can form the new camera matrix for
     * each view where the principal points are located at the center.
     *
     * @param cameraMatrix Input camera matrix.
     * @param imgsize Camera view image size in pixels.
     * @param centerPrincipalPoint Location of the principal point in the new camera matrix. The
     * parameter indicates whether this location should be at the image center or not.
     * @return automatically generated
     */
    public static Mat getDefaultNewCameraMatrix(Mat cameraMatrix, Size imgsize, boolean centerPrincipalPoint) {
        return new Mat(getDefaultNewCameraMatrix_0(cameraMatrix.nativeObj, imgsize.width, imgsize.height, centerPrincipalPoint));
    }

    /**
     * Returns the default new camera matrix.
     *
     * The function returns the camera matrix that is either an exact copy of the input cameraMatrix (when
     * centerPrinicipalPoint=false ), or the modified one (when centerPrincipalPoint=true).
     *
     * In the latter case, the new camera matrix will be:
     *
     * \(\begin{bmatrix} f_x &amp;&amp; 0 &amp;&amp; ( \texttt{imgSize.width} -1)*0.5  \\ 0 &amp;&amp; f_y &amp;&amp; ( \texttt{imgSize.height} -1)*0.5  \\ 0 &amp;&amp; 0 &amp;&amp; 1 \end{bmatrix} ,\)
     *
     * where \(f_x\) and \(f_y\) are \((0,0)\) and \((1,1)\) elements of cameraMatrix, respectively.
     *
     * By default, the undistortion functions in OpenCV (see #initUndistortRectifyMap, #undistort) do not
     * move the principal point. However, when you work with stereo, it is important to move the principal
     * points in both views to the same y-coordinate (which is required by most of stereo correspondence
     * algorithms), and may be to the same x-coordinate too. So, you can form the new camera matrix for
     * each view where the principal points are located at the center.
     *
     * @param cameraMatrix Input camera matrix.
     * @param imgsize Camera view image size in pixels.
     * parameter indicates whether this location should be at the image center or not.
     * @return automatically generated
     */
    public static Mat getDefaultNewCameraMatrix(Mat cameraMatrix, Size imgsize) {
        return new Mat(getDefaultNewCameraMatrix_1(cameraMatrix.nativeObj, imgsize.width, imgsize.height));
    }

    /**
     * Returns the default new camera matrix.
     *
     * The function returns the camera matrix that is either an exact copy of the input cameraMatrix (when
     * centerPrinicipalPoint=false ), or the modified one (when centerPrincipalPoint=true).
     *
     * In the latter case, the new camera matrix will be:
     *
     * \(\begin{bmatrix} f_x &amp;&amp; 0 &amp;&amp; ( \texttt{imgSize.width} -1)*0.5  \\ 0 &amp;&amp; f_y &amp;&amp; ( \texttt{imgSize.height} -1)*0.5  \\ 0 &amp;&amp; 0 &amp;&amp; 1 \end{bmatrix} ,\)
     *
     * where \(f_x\) and \(f_y\) are \((0,0)\) and \((1,1)\) elements of cameraMatrix, respectively.
     *
     * By default, the undistortion functions in OpenCV (see #initUndistortRectifyMap, #undistort) do not
     * move the principal point. However, when you work with stereo, it is important to move the principal
     * points in both views to the same y-coordinate (which is required by most of stereo correspondence
     * algorithms), and may be to the same x-coordinate too. So, you can form the new camera matrix for
     * each view where the principal points are located at the center.
     *
     * @param cameraMatrix Input camera matrix.
     * parameter indicates whether this location should be at the image center or not.
     * @return automatically generated
     */
    public static Mat getDefaultNewCameraMatrix(Mat cameraMatrix) {
        return new Mat(getDefaultNewCameraMatrix_2(cameraMatrix.nativeObj));
    }


    //
    // C++:  Mat cv::getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha, Size newImgSize = Size(), Rect* validPixROI = 0, bool centerPrincipalPoint = false)
    //

    /**
     * Returns the new camera matrix based on the free scaling parameter.
     *
     * @param cameraMatrix Input camera matrix.
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param imageSize Original image size.
     * @param alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are
     * valid) and 1 (when all the source image pixels are retained in the undistorted image). See
     * stereoRectify for details.
     * @param newImgSize Image size after rectification. By default, it is set to imageSize .
     * @param validPixROI Optional output rectangle that outlines all-good-pixels region in the
     * undistorted image. See roi1, roi2 description in stereoRectify .
     * @param centerPrincipalPoint Optional flag that indicates whether in the new camera matrix the
     * principal point should be at the image center or not. By default, the principal point is chosen to
     * best fit a subset of the source image (determined by alpha) to the corrected image.
     * @return new_camera_matrix Output new camera matrix.
     *
     * The function computes and returns the optimal new camera matrix based on the free scaling parameter.
     * By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original
     * image pixels if there is valuable information in the corners alpha=1 , or get something in between.
     * When alpha&gt;0 , the undistorted result is likely to have some black pixels corresponding to
     * "virtual" pixels outside of the captured distorted image. The original camera matrix, distortion
     * coefficients, the computed new camera matrix, and newImageSize should be passed to
     * initUndistortRectifyMap to produce the maps for remap .
     */
    public static Mat getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha, Size newImgSize, Rect validPixROI, boolean centerPrincipalPoint) {
        double[] validPixROI_out = new double[4];
        Mat retVal = new Mat(getOptimalNewCameraMatrix_0(cameraMatrix.nativeObj, distCoeffs.nativeObj, imageSize.width, imageSize.height, alpha, newImgSize.width, newImgSize.height, validPixROI_out, centerPrincipalPoint));
        if(validPixROI!=null){ validPixROI.x = (int)validPixROI_out[0]; validPixROI.y = (int)validPixROI_out[1]; validPixROI.width = (int)validPixROI_out[2]; validPixROI.height = (int)validPixROI_out[3]; } 
        return retVal;
    }

    /**
     * Returns the new camera matrix based on the free scaling parameter.
     *
     * @param cameraMatrix Input camera matrix.
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param imageSize Original image size.
     * @param alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are
     * valid) and 1 (when all the source image pixels are retained in the undistorted image). See
     * stereoRectify for details.
     * @param newImgSize Image size after rectification. By default, it is set to imageSize .
     * @param validPixROI Optional output rectangle that outlines all-good-pixels region in the
     * undistorted image. See roi1, roi2 description in stereoRectify .
     * principal point should be at the image center or not. By default, the principal point is chosen to
     * best fit a subset of the source image (determined by alpha) to the corrected image.
     * @return new_camera_matrix Output new camera matrix.
     *
     * The function computes and returns the optimal new camera matrix based on the free scaling parameter.
     * By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original
     * image pixels if there is valuable information in the corners alpha=1 , or get something in between.
     * When alpha&gt;0 , the undistorted result is likely to have some black pixels corresponding to
     * "virtual" pixels outside of the captured distorted image. The original camera matrix, distortion
     * coefficients, the computed new camera matrix, and newImageSize should be passed to
     * initUndistortRectifyMap to produce the maps for remap .
     */
    public static Mat getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha, Size newImgSize, Rect validPixROI) {
        double[] validPixROI_out = new double[4];
        Mat retVal = new Mat(getOptimalNewCameraMatrix_1(cameraMatrix.nativeObj, distCoeffs.nativeObj, imageSize.width, imageSize.height, alpha, newImgSize.width, newImgSize.height, validPixROI_out));
        if(validPixROI!=null){ validPixROI.x = (int)validPixROI_out[0]; validPixROI.y = (int)validPixROI_out[1]; validPixROI.width = (int)validPixROI_out[2]; validPixROI.height = (int)validPixROI_out[3]; } 
        return retVal;
    }

    /**
     * Returns the new camera matrix based on the free scaling parameter.
     *
     * @param cameraMatrix Input camera matrix.
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param imageSize Original image size.
     * @param alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are
     * valid) and 1 (when all the source image pixels are retained in the undistorted image). See
     * stereoRectify for details.
     * @param newImgSize Image size after rectification. By default, it is set to imageSize .
     * undistorted image. See roi1, roi2 description in stereoRectify .
     * principal point should be at the image center or not. By default, the principal point is chosen to
     * best fit a subset of the source image (determined by alpha) to the corrected image.
     * @return new_camera_matrix Output new camera matrix.
     *
     * The function computes and returns the optimal new camera matrix based on the free scaling parameter.
     * By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original
     * image pixels if there is valuable information in the corners alpha=1 , or get something in between.
     * When alpha&gt;0 , the undistorted result is likely to have some black pixels corresponding to
     * "virtual" pixels outside of the captured distorted image. The original camera matrix, distortion
     * coefficients, the computed new camera matrix, and newImageSize should be passed to
     * initUndistortRectifyMap to produce the maps for remap .
     */
    public static Mat getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha, Size newImgSize) {
        return new Mat(getOptimalNewCameraMatrix_2(cameraMatrix.nativeObj, distCoeffs.nativeObj, imageSize.width, imageSize.height, alpha, newImgSize.width, newImgSize.height));
    }

    /**
     * Returns the new camera matrix based on the free scaling parameter.
     *
     * @param cameraMatrix Input camera matrix.
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param imageSize Original image size.
     * @param alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are
     * valid) and 1 (when all the source image pixels are retained in the undistorted image). See
     * stereoRectify for details.
     * undistorted image. See roi1, roi2 description in stereoRectify .
     * principal point should be at the image center or not. By default, the principal point is chosen to
     * best fit a subset of the source image (determined by alpha) to the corrected image.
     * @return new_camera_matrix Output new camera matrix.
     *
     * The function computes and returns the optimal new camera matrix based on the free scaling parameter.
     * By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original
     * image pixels if there is valuable information in the corners alpha=1 , or get something in between.
     * When alpha&gt;0 , the undistorted result is likely to have some black pixels corresponding to
     * "virtual" pixels outside of the captured distorted image. The original camera matrix, distortion
     * coefficients, the computed new camera matrix, and newImageSize should be passed to
     * initUndistortRectifyMap to produce the maps for remap .
     */
    public static Mat getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha) {
        return new Mat(getOptimalNewCameraMatrix_3(cameraMatrix.nativeObj, distCoeffs.nativeObj, imageSize.width, imageSize.height, alpha));
    }


    //
    // C++:  Mat cv::initCameraMatrix2D(vector_vector_Point3f objectPoints, vector_vector_Point2f imagePoints, Size imageSize, double aspectRatio = 1.0)
    //

    /**
     * Finds an initial camera matrix from 3D-2D point correspondences.
     *
     * @param objectPoints Vector of vectors of the calibration pattern points in the calibration pattern
     * coordinate space. In the old interface all the per-view vectors are concatenated. See
     * calibrateCamera for details.
     * @param imagePoints Vector of vectors of the projections of the calibration pattern points. In the
     * old interface all the per-view vectors are concatenated.
     * @param imageSize Image size in pixels used to initialize the principal point.
     * @param aspectRatio If it is zero or negative, both \(f_x\) and \(f_y\) are estimated independently.
     * Otherwise, \(f_x = f_y * \texttt{aspectRatio}\) .
     *
     * The function estimates and returns an initial camera matrix for the camera calibration process.
     * Currently, the function only supports planar calibration patterns, which are patterns where each
     * object point has z-coordinate =0.
     * @return automatically generated
     */
    public static Mat initCameraMatrix2D(List<MatOfPoint3f> objectPoints, List<MatOfPoint2f> imagePoints, Size imageSize, double aspectRatio) {
        List<Mat> objectPoints_tmplm = new ArrayList<Mat>((objectPoints != null) ? objectPoints.size() : 0);
        Mat objectPoints_mat = Converters.vector_vector_Point3f_to_Mat(objectPoints, objectPoints_tmplm);
        List<Mat> imagePoints_tmplm = new ArrayList<Mat>((imagePoints != null) ? imagePoints.size() : 0);
        Mat imagePoints_mat = Converters.vector_vector_Point2f_to_Mat(imagePoints, imagePoints_tmplm);
        return new Mat(initCameraMatrix2D_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, aspectRatio));
    }

    /**
     * Finds an initial camera matrix from 3D-2D point correspondences.
     *
     * @param objectPoints Vector of vectors of the calibration pattern points in the calibration pattern
     * coordinate space. In the old interface all the per-view vectors are concatenated. See
     * calibrateCamera for details.
     * @param imagePoints Vector of vectors of the projections of the calibration pattern points. In the
     * old interface all the per-view vectors are concatenated.
     * @param imageSize Image size in pixels used to initialize the principal point.
     * Otherwise, \(f_x = f_y * \texttt{aspectRatio}\) .
     *
     * The function estimates and returns an initial camera matrix for the camera calibration process.
     * Currently, the function only supports planar calibration patterns, which are patterns where each
     * object point has z-coordinate =0.
     * @return automatically generated
     */
    public static Mat initCameraMatrix2D(List<MatOfPoint3f> objectPoints, List<MatOfPoint2f> imagePoints, Size imageSize) {
        List<Mat> objectPoints_tmplm = new ArrayList<Mat>((objectPoints != null) ? objectPoints.size() : 0);
        Mat objectPoints_mat = Converters.vector_vector_Point3f_to_Mat(objectPoints, objectPoints_tmplm);
        List<Mat> imagePoints_tmplm = new ArrayList<Mat>((imagePoints != null) ? imagePoints.size() : 0);
        Mat imagePoints_mat = Converters.vector_vector_Point2f_to_Mat(imagePoints, imagePoints_tmplm);
        return new Mat(initCameraMatrix2D_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height));
    }


    //
    // C++:  Rect cv::getValidDisparityROI(Rect roi1, Rect roi2, int minDisparity, int numberOfDisparities, int blockSize)
    //

    public static Rect getValidDisparityROI(Rect roi1, Rect roi2, int minDisparity, int numberOfDisparities, int blockSize) {
        return new Rect(getValidDisparityROI_0(roi1.x, roi1.y, roi1.width, roi1.height, roi2.x, roi2.y, roi2.width, roi2.height, minDisparity, numberOfDisparities, blockSize));
    }


    //
    // C++:  Scalar cv::estimateChessboardSharpness(Mat image, Size patternSize, Mat corners, float rise_distance = 0.8F, bool vertical = false, Mat& sharpness = Mat())
    //

    /**
     * Estimates the sharpness of a detected chessboard.
     *
     * Image sharpness, as well as brightness, are a critical parameter for accuracte
     * camera calibration. For accessing these parameters for filtering out
     * problematic calibraiton images, this method calculates edge profiles by traveling from
     * black to white chessboard cell centers. Based on this, the number of pixels is
     * calculated required to transit from black to white. This width of the
     * transition area is a good indication of how sharp the chessboard is imaged
     * and should be below ~3.0 pixels.
     *
     * @param image Gray image used to find chessboard corners
     * @param patternSize Size of a found chessboard pattern
     * @param corners Corners found by findChessboardCorners(SB)
     * @param rise_distance Rise distance 0.8 means 10% ... 90% of the final signal strength
     * @param vertical By default edge responses for horizontal lines are calculated
     * @param sharpness Optional output array with a sharpness value for calculated edge responses (see description)
     *
     * The optional sharpness array is of type CV_32FC1 and has for each calculated
     * profile one row with the following five entries:
     * 0 = x coordinate of the underlying edge in the image
     * 1 = y coordinate of the underlying edge in the image
     * 2 = width of the transition area (sharpness)
     * 3 = signal strength in the black cell (min brightness)
     * 4 = signal strength in the white cell (max brightness)
     *
     * @return Scalar(average sharpness, average min brightness, average max brightness,0)
     */
    public static Scalar estimateChessboardSharpness(Mat image, Size patternSize, Mat corners, float rise_distance, boolean vertical, Mat sharpness) {
        return new Scalar(estimateChessboardSharpness_0(image.nativeObj, patternSize.width, patternSize.height, corners.nativeObj, rise_distance, vertical, sharpness.nativeObj));
    }

    /**
     * Estimates the sharpness of a detected chessboard.
     *
     * Image sharpness, as well as brightness, are a critical parameter for accuracte
     * camera calibration. For accessing these parameters for filtering out
     * problematic calibraiton images, this method calculates edge profiles by traveling from
     * black to white chessboard cell centers. Based on this, the number of pixels is
     * calculated required to transit from black to white. This width of the
     * transition area is a good indication of how sharp the chessboard is imaged
     * and should be below ~3.0 pixels.
     *
     * @param image Gray image used to find chessboard corners
     * @param patternSize Size of a found chessboard pattern
     * @param corners Corners found by findChessboardCorners(SB)
     * @param rise_distance Rise distance 0.8 means 10% ... 90% of the final signal strength
     * @param vertical By default edge responses for horizontal lines are calculated
     *
     * The optional sharpness array is of type CV_32FC1 and has for each calculated
     * profile one row with the following five entries:
     * 0 = x coordinate of the underlying edge in the image
     * 1 = y coordinate of the underlying edge in the image
     * 2 = width of the transition area (sharpness)
     * 3 = signal strength in the black cell (min brightness)
     * 4 = signal strength in the white cell (max brightness)
     *
     * @return Scalar(average sharpness, average min brightness, average max brightness,0)
     */
    public static Scalar estimateChessboardSharpness(Mat image, Size patternSize, Mat corners, float rise_distance, boolean vertical) {
        return new Scalar(estimateChessboardSharpness_1(image.nativeObj, patternSize.width, patternSize.height, corners.nativeObj, rise_distance, vertical));
    }

    /**
     * Estimates the sharpness of a detected chessboard.
     *
     * Image sharpness, as well as brightness, are a critical parameter for accuracte
     * camera calibration. For accessing these parameters for filtering out
     * problematic calibraiton images, this method calculates edge profiles by traveling from
     * black to white chessboard cell centers. Based on this, the number of pixels is
     * calculated required to transit from black to white. This width of the
     * transition area is a good indication of how sharp the chessboard is imaged
     * and should be below ~3.0 pixels.
     *
     * @param image Gray image used to find chessboard corners
     * @param patternSize Size of a found chessboard pattern
     * @param corners Corners found by findChessboardCorners(SB)
     * @param rise_distance Rise distance 0.8 means 10% ... 90% of the final signal strength
     *
     * The optional sharpness array is of type CV_32FC1 and has for each calculated
     * profile one row with the following five entries:
     * 0 = x coordinate of the underlying edge in the image
     * 1 = y coordinate of the underlying edge in the image
     * 2 = width of the transition area (sharpness)
     * 3 = signal strength in the black cell (min brightness)
     * 4 = signal strength in the white cell (max brightness)
     *
     * @return Scalar(average sharpness, average min brightness, average max brightness,0)
     */
    public static Scalar estimateChessboardSharpness(Mat image, Size patternSize, Mat corners, float rise_distance) {
        return new Scalar(estimateChessboardSharpness_2(image.nativeObj, patternSize.width, patternSize.height, corners.nativeObj, rise_distance));
    }

    /**
     * Estimates the sharpness of a detected chessboard.
     *
     * Image sharpness, as well as brightness, are a critical parameter for accuracte
     * camera calibration. For accessing these parameters for filtering out
     * problematic calibraiton images, this method calculates edge profiles by traveling from
     * black to white chessboard cell centers. Based on this, the number of pixels is
     * calculated required to transit from black to white. This width of the
     * transition area is a good indication of how sharp the chessboard is imaged
     * and should be below ~3.0 pixels.
     *
     * @param image Gray image used to find chessboard corners
     * @param patternSize Size of a found chessboard pattern
     * @param corners Corners found by findChessboardCorners(SB)
     *
     * The optional sharpness array is of type CV_32FC1 and has for each calculated
     * profile one row with the following five entries:
     * 0 = x coordinate of the underlying edge in the image
     * 1 = y coordinate of the underlying edge in the image
     * 2 = width of the transition area (sharpness)
     * 3 = signal strength in the black cell (min brightness)
     * 4 = signal strength in the white cell (max brightness)
     *
     * @return Scalar(average sharpness, average min brightness, average max brightness,0)
     */
    public static Scalar estimateChessboardSharpness(Mat image, Size patternSize, Mat corners) {
        return new Scalar(estimateChessboardSharpness_3(image.nativeObj, patternSize.width, patternSize.height, corners.nativeObj));
    }


    //
    // C++:  Vec3d cv::RQDecomp3x3(Mat src, Mat& mtxR, Mat& mtxQ, Mat& Qx = Mat(), Mat& Qy = Mat(), Mat& Qz = Mat())
    //

    /**
     * Computes an RQ decomposition of 3x3 matrices.
     *
     * @param src 3x3 input matrix.
     * @param mtxR Output 3x3 upper-triangular matrix.
     * @param mtxQ Output 3x3 orthogonal matrix.
     * @param Qx Optional output 3x3 rotation matrix around x-axis.
     * @param Qy Optional output 3x3 rotation matrix around y-axis.
     * @param Qz Optional output 3x3 rotation matrix around z-axis.
     *
     * The function computes a RQ decomposition using the given rotations. This function is used in
     * decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera
     * and a rotation matrix.
     *
     * It optionally returns three rotation matrices, one for each axis, and the three Euler angles in
     * degrees (as the return value) that could be used in OpenGL. Note, there is always more than one
     * sequence of rotations about the three principal axes that results in the same orientation of an
     * object, e.g. see CITE: Slabaugh . Returned tree rotation matrices and corresponding three Euler angles
     * are only one of the possible solutions.
     * @return automatically generated
     */
    public static double[] RQDecomp3x3(Mat src, Mat mtxR, Mat mtxQ, Mat Qx, Mat Qy, Mat Qz) {
        return RQDecomp3x3_0(src.nativeObj, mtxR.nativeObj, mtxQ.nativeObj, Qx.nativeObj, Qy.nativeObj, Qz.nativeObj);
    }

    /**
     * Computes an RQ decomposition of 3x3 matrices.
     *
     * @param src 3x3 input matrix.
     * @param mtxR Output 3x3 upper-triangular matrix.
     * @param mtxQ Output 3x3 orthogonal matrix.
     * @param Qx Optional output 3x3 rotation matrix around x-axis.
     * @param Qy Optional output 3x3 rotation matrix around y-axis.
     *
     * The function computes a RQ decomposition using the given rotations. This function is used in
     * decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera
     * and a rotation matrix.
     *
     * It optionally returns three rotation matrices, one for each axis, and the three Euler angles in
     * degrees (as the return value) that could be used in OpenGL. Note, there is always more than one
     * sequence of rotations about the three principal axes that results in the same orientation of an
     * object, e.g. see CITE: Slabaugh . Returned tree rotation matrices and corresponding three Euler angles
     * are only one of the possible solutions.
     * @return automatically generated
     */
    public static double[] RQDecomp3x3(Mat src, Mat mtxR, Mat mtxQ, Mat Qx, Mat Qy) {
        return RQDecomp3x3_1(src.nativeObj, mtxR.nativeObj, mtxQ.nativeObj, Qx.nativeObj, Qy.nativeObj);
    }

    /**
     * Computes an RQ decomposition of 3x3 matrices.
     *
     * @param src 3x3 input matrix.
     * @param mtxR Output 3x3 upper-triangular matrix.
     * @param mtxQ Output 3x3 orthogonal matrix.
     * @param Qx Optional output 3x3 rotation matrix around x-axis.
     *
     * The function computes a RQ decomposition using the given rotations. This function is used in
     * decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera
     * and a rotation matrix.
     *
     * It optionally returns three rotation matrices, one for each axis, and the three Euler angles in
     * degrees (as the return value) that could be used in OpenGL. Note, there is always more than one
     * sequence of rotations about the three principal axes that results in the same orientation of an
     * object, e.g. see CITE: Slabaugh . Returned tree rotation matrices and corresponding three Euler angles
     * are only one of the possible solutions.
     * @return automatically generated
     */
    public static double[] RQDecomp3x3(Mat src, Mat mtxR, Mat mtxQ, Mat Qx) {
        return RQDecomp3x3_2(src.nativeObj, mtxR.nativeObj, mtxQ.nativeObj, Qx.nativeObj);
    }

    /**
     * Computes an RQ decomposition of 3x3 matrices.
     *
     * @param src 3x3 input matrix.
     * @param mtxR Output 3x3 upper-triangular matrix.
     * @param mtxQ Output 3x3 orthogonal matrix.
     *
     * The function computes a RQ decomposition using the given rotations. This function is used in
     * decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera
     * and a rotation matrix.
     *
     * It optionally returns three rotation matrices, one for each axis, and the three Euler angles in
     * degrees (as the return value) that could be used in OpenGL. Note, there is always more than one
     * sequence of rotations about the three principal axes that results in the same orientation of an
     * object, e.g. see CITE: Slabaugh . Returned tree rotation matrices and corresponding three Euler angles
     * are only one of the possible solutions.
     * @return automatically generated
     */
    public static double[] RQDecomp3x3(Mat src, Mat mtxR, Mat mtxQ) {
        return RQDecomp3x3_3(src.nativeObj, mtxR.nativeObj, mtxQ.nativeObj);
    }


    //
    // C++:  bool cv::checkChessboard(Mat img, Size size)
    //

    public static boolean checkChessboard(Mat img, Size size) {
        return checkChessboard_0(img.nativeObj, size.width, size.height);
    }


    //
    // C++:  bool cv::find4QuadCornerSubpix(Mat img, Mat& corners, Size region_size)
    //

    public static boolean find4QuadCornerSubpix(Mat img, Mat corners, Size region_size) {
        return find4QuadCornerSubpix_0(img.nativeObj, corners.nativeObj, region_size.width, region_size.height);
    }


    //
    // C++:  bool cv::findChessboardCorners(Mat image, Size patternSize, vector_Point2f& corners, int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE)
    //

    /**
     * Finds the positions of internal corners of the chessboard.
     *
     * @param image Source chessboard view. It must be an 8-bit grayscale or color image.
     * @param patternSize Number of inner corners per a chessboard row and column
     * ( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
     * @param corners Output array of detected corners.
     * @param flags Various operation flags that can be zero or a combination of the following values:
     * <ul>
     *   <li>
     *    <b>CALIB_CB_ADAPTIVE_THRESH</b> Use adaptive thresholding to convert the image to black
     * and white, rather than a fixed threshold level (computed from the average image brightness).
     *   </li>
     *   <li>
     *    <b>CALIB_CB_NORMALIZE_IMAGE</b> Normalize the image gamma with equalizeHist before
     * applying fixed or adaptive thresholding.
     *   </li>
     *   <li>
     *    <b>CALIB_CB_FILTER_QUADS</b> Use additional criteria (like contour area, perimeter,
     * square-like shape) to filter out false quads extracted at the contour retrieval stage.
     *   </li>
     *   <li>
     *    <b>CALIB_CB_FAST_CHECK</b> Run a fast check on the image that looks for chessboard corners,
     * and shortcut the call if none is found. This can drastically speed up the call in the
     * degenerate condition when no chessboard is observed.
     *   </li>
     * </ul>
     *
     * The function attempts to determine whether the input image is a view of the chessboard pattern and
     * locate the internal chessboard corners. The function returns a non-zero value if all of the corners
     * are found and they are placed in a certain order (row by row, left to right in every row).
     * Otherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,
     * a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black
     * squares touch each other. The detected coordinates are approximate, and to determine their positions
     * more accurately, the function calls cornerSubPix. You also may use the function cornerSubPix with
     * different parameters if returned coordinates are not accurate enough.
     *
     * Sample usage of detecting and drawing chessboard corners: :
     * <code>
     *     Size patternsize(8,6); //interior number of corners
     *     Mat gray = ....; //source image
     *     vector&lt;Point2f&gt; corners; //this will be filled by the detected corners
     *
     *     //CALIB_CB_FAST_CHECK saves a lot of time on images
     *     //that do not contain any chessboard corners
     *     bool patternfound = findChessboardCorners(gray, patternsize, corners,
     *             CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
     *             + CALIB_CB_FAST_CHECK);
     *
     *     if(patternfound)
     *       cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
     *         TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
     *
     *     drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
     * </code>
     * <b>Note:</b> The function requires white space (like a square-thick border, the wider the better) around
     * the board to make the detection more robust in various environments. Otherwise, if there is no
     * border and the background is dark, the outer black squares cannot be segmented properly and so the
     * square grouping and ordering algorithm fails.
     * @return automatically generated
     */
    public static boolean findChessboardCorners(Mat image, Size patternSize, MatOfPoint2f corners, int flags) {
        Mat corners_mat = corners;
        return findChessboardCorners_0(image.nativeObj, patternSize.width, patternSize.height, corners_mat.nativeObj, flags);
    }

    /**
     * Finds the positions of internal corners of the chessboard.
     *
     * @param image Source chessboard view. It must be an 8-bit grayscale or color image.
     * @param patternSize Number of inner corners per a chessboard row and column
     * ( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
     * @param corners Output array of detected corners.
     * <ul>
     *   <li>
     *    <b>CALIB_CB_ADAPTIVE_THRESH</b> Use adaptive thresholding to convert the image to black
     * and white, rather than a fixed threshold level (computed from the average image brightness).
     *   </li>
     *   <li>
     *    <b>CALIB_CB_NORMALIZE_IMAGE</b> Normalize the image gamma with equalizeHist before
     * applying fixed or adaptive thresholding.
     *   </li>
     *   <li>
     *    <b>CALIB_CB_FILTER_QUADS</b> Use additional criteria (like contour area, perimeter,
     * square-like shape) to filter out false quads extracted at the contour retrieval stage.
     *   </li>
     *   <li>
     *    <b>CALIB_CB_FAST_CHECK</b> Run a fast check on the image that looks for chessboard corners,
     * and shortcut the call if none is found. This can drastically speed up the call in the
     * degenerate condition when no chessboard is observed.
     *   </li>
     * </ul>
     *
     * The function attempts to determine whether the input image is a view of the chessboard pattern and
     * locate the internal chessboard corners. The function returns a non-zero value if all of the corners
     * are found and they are placed in a certain order (row by row, left to right in every row).
     * Otherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,
     * a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black
     * squares touch each other. The detected coordinates are approximate, and to determine their positions
     * more accurately, the function calls cornerSubPix. You also may use the function cornerSubPix with
     * different parameters if returned coordinates are not accurate enough.
     *
     * Sample usage of detecting and drawing chessboard corners: :
     * <code>
     *     Size patternsize(8,6); //interior number of corners
     *     Mat gray = ....; //source image
     *     vector&lt;Point2f&gt; corners; //this will be filled by the detected corners
     *
     *     //CALIB_CB_FAST_CHECK saves a lot of time on images
     *     //that do not contain any chessboard corners
     *     bool patternfound = findChessboardCorners(gray, patternsize, corners,
     *             CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
     *             + CALIB_CB_FAST_CHECK);
     *
     *     if(patternfound)
     *       cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
     *         TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
     *
     *     drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
     * </code>
     * <b>Note:</b> The function requires white space (like a square-thick border, the wider the better) around
     * the board to make the detection more robust in various environments. Otherwise, if there is no
     * border and the background is dark, the outer black squares cannot be segmented properly and so the
     * square grouping and ordering algorithm fails.
     * @return automatically generated
     */
    public static boolean findChessboardCorners(Mat image, Size patternSize, MatOfPoint2f corners) {
        Mat corners_mat = corners;
        return findChessboardCorners_1(image.nativeObj, patternSize.width, patternSize.height, corners_mat.nativeObj);
    }


    //
    // C++:  bool cv::findChessboardCornersSB(Mat image, Size patternSize, Mat& corners, int flags, Mat& meta)
    //

    /**
     * Finds the positions of internal corners of the chessboard using a sector based approach.
     *
     * @param image Source chessboard view. It must be an 8-bit grayscale or color image.
     * @param patternSize Number of inner corners per a chessboard row and column
     * ( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
     * @param corners Output array of detected corners.
     * @param flags Various operation flags that can be zero or a combination of the following values:
     * <ul>
     *   <li>
     *    <b>CALIB_CB_NORMALIZE_IMAGE</b> Normalize the image gamma with equalizeHist before detection.
     *   </li>
     *   <li>
     *    <b>CALIB_CB_EXHAUSTIVE</b> Run an exhaustive search to improve detection rate.
     *   </li>
     *   <li>
     *    <b>CALIB_CB_ACCURACY</b> Up sample input image to improve sub-pixel accuracy due to aliasing effects.
     *   </li>
     *   <li>
     *    <b>CALIB_CB_LARGER</b> The detected pattern is allowed to be larger than patternSize (see description).
     *   </li>
     *   <li>
     *    <b>CALIB_CB_MARKER</b> The detected pattern must have a marker (see description).
     * This should be used if an accurate camera calibration is required.
     * @param meta Optional output arrray of detected corners (CV_8UC1 and size = cv::Size(columns,rows)).
     * Each entry stands for one corner of the pattern and can have one of the following values:
     *   </li>
     *   <li>
     *    0 = no meta data attached
     *   </li>
     *   <li>
     *    1 = left-top corner of a black cell
     *   </li>
     *   <li>
     *    2 = left-top corner of a white cell
     *   </li>
     *   <li>
     *    3 = left-top corner of a black cell with a white marker dot
     *   </li>
     *   <li>
     *    4 = left-top corner of a white cell with a black marker dot (pattern origin in case of markers otherwise first corner)
     *   </li>
     * </ul>
     *
     * The function is analog to findchessboardCorners but uses a localized radon
     * transformation approximated by box filters being more robust to all sort of
     * noise, faster on larger images and is able to directly return the sub-pixel
     * position of the internal chessboard corners. The Method is based on the paper
     * CITE: duda2018 "Accurate Detection and Localization of Checkerboard Corners for
     * Calibration" demonstrating that the returned sub-pixel positions are more
     * accurate than the one returned by cornerSubPix allowing a precise camera
     * calibration for demanding applications.
     *
     * In the case, the flags <b>CALIB_CB_LARGER</b> or <b>CALIB_CB_MARKER</b> are given,
     * the result can be recovered from the optional meta array. Both flags are
     * helpful to use calibration patterns exceeding the field of view of the camera.
     * These oversized patterns allow more accurate calibrations as corners can be
     * utilized, which are as close as possible to the image borders.  For a
     * consistent coordinate system across all images, the optional marker (see image
     * below) can be used to move the origin of the board to the location where the
     * black circle is located.
     *
     * <b>Note:</b> The function requires a white boarder with roughly the same width as one
     * of the checkerboard fields around the whole board to improve the detection in
     * various environments. In addition, because of the localized radon
     * transformation it is beneficial to use round corners for the field corners
     * which are located on the outside of the board. The following figure illustrates
     * a sample checkerboard optimized for the detection. However, any other checkerboard
     * can be used as well.
     * ![Checkerboard](pics/checkerboard_radon.png)
     * @return automatically generated
     */
    public static boolean findChessboardCornersSBWithMeta(Mat image, Size patternSize, Mat corners, int flags, Mat meta) {
        return findChessboardCornersSBWithMeta_0(image.nativeObj, patternSize.width, patternSize.height, corners.nativeObj, flags, meta.nativeObj);
    }


    //
    // C++:  bool cv::findChessboardCornersSB(Mat image, Size patternSize, Mat& corners, int flags = 0)
    //

    public static boolean findChessboardCornersSB(Mat image, Size patternSize, Mat corners, int flags) {
        return findChessboardCornersSB_0(image.nativeObj, patternSize.width, patternSize.height, corners.nativeObj, flags);
    }

    public static boolean findChessboardCornersSB(Mat image, Size patternSize, Mat corners) {
        return findChessboardCornersSB_1(image.nativeObj, patternSize.width, patternSize.height, corners.nativeObj);
    }


    //
    // C++:  bool cv::findCirclesGrid(Mat image, Size patternSize, Mat& centers, int flags, Ptr_FeatureDetector blobDetector, CirclesGridFinderParameters parameters)
    //

    // Unknown type 'Ptr_FeatureDetector' (I), skipping the function


    //
    // C++:  bool cv::findCirclesGrid(Mat image, Size patternSize, Mat& centers, int flags = CALIB_CB_SYMMETRIC_GRID, Ptr_FeatureDetector blobDetector = SimpleBlobDetector::create())
    //

    public static boolean findCirclesGrid(Mat image, Size patternSize, Mat centers, int flags) {
        return findCirclesGrid_0(image.nativeObj, patternSize.width, patternSize.height, centers.nativeObj, flags);
    }

    public static boolean findCirclesGrid(Mat image, Size patternSize, Mat centers) {
        return findCirclesGrid_2(image.nativeObj, patternSize.width, patternSize.height, centers.nativeObj);
    }


    //
    // C++:  bool cv::solvePnP(vector_Point3f objectPoints, vector_Point2f imagePoints, Mat cameraMatrix, vector_double distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE)
    //

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns the rotation and the translation vectors that transform a 3D point expressed in the object
     * coordinate frame to the camera coordinate frame, using different methods:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): need 4 input points to return a unique solution.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * @param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param flags Method for solving a PnP problem:
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F. Moreno-Noguer, V. Lepetit and P. Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of J. Hesch and S. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A. Penate-Sanchez, J. Andrade-Cetto,
     * F. Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnP(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess, int flags) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnP_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess, flags);
    }

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns the rotation and the translation vectors that transform a 3D point expressed in the object
     * coordinate frame to the camera coordinate frame, using different methods:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): need 4 input points to return a unique solution.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * @param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F. Moreno-Noguer, V. Lepetit and P. Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of J. Hesch and S. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A. Penate-Sanchez, J. Andrade-Cetto,
     * F. Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnP(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnP_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess);
    }

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns the rotation and the translation vectors that transform a 3D point expressed in the object
     * coordinate frame to the camera coordinate frame, using different methods:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): need 4 input points to return a unique solution.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F. Moreno-Noguer, V. Lepetit and P. Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of J. Hesch and S. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A. Penate-Sanchez, J. Andrade-Cetto,
     * F. Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnP(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnP_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj);
    }


    //
    // C++:  bool cv::solvePnPRansac(vector_Point3f objectPoints, vector_Point2f imagePoints, Mat cameraMatrix, vector_double distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess = false, int iterationsCount = 100, float reprojectionError = 8.0, double confidence = 0.99, Mat& inliers = Mat(), int flags = SOLVEPNP_ITERATIVE)
    //

    /**
     * Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * @param useExtrinsicGuess Parameter used for REF: SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param iterationsCount Number of iterations.
     * @param reprojectionError Inlier threshold value used by the RANSAC procedure. The parameter value
     * is the maximum allowed distance between the observed and computed point projections to consider it
     * an inlier.
     * @param confidence The probability that the algorithm produces a useful result.
     * @param inliers Output vector that contains indices of inliers in objectPoints and imagePoints .
     * @param flags Method for solving a PnP problem (see REF: solvePnP ).
     *
     * The function estimates an object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients. This function finds such
     * a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
     * projections imagePoints and the projected (using REF: projectPoints ) objectPoints. The use of RANSAC
     * makes the function resistant to outliers.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePNPRansac for object detection can be found at
     *         opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
     *   </li>
     *   <li>
     *       The default method used to estimate the camera pose for the Minimal Sample Sets step
     *        is #SOLVEPNP_EPNP. Exceptions are:
     *   <ul>
     *     <li>
     *           if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
     *     </li>
     *     <li>
     *           if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
     *     </li>
     *   </ul>
     *   <li>
     *       The method used to estimate the camera pose using all the inliers is defined by the
     *        flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
     *        the method #SOLVEPNP_EPNP will be used instead.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError, double confidence, Mat inliers, int flags) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnPRansac_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers.nativeObj, flags);
    }

    /**
     * Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * @param useExtrinsicGuess Parameter used for REF: SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param iterationsCount Number of iterations.
     * @param reprojectionError Inlier threshold value used by the RANSAC procedure. The parameter value
     * is the maximum allowed distance between the observed and computed point projections to consider it
     * an inlier.
     * @param confidence The probability that the algorithm produces a useful result.
     * @param inliers Output vector that contains indices of inliers in objectPoints and imagePoints .
     *
     * The function estimates an object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients. This function finds such
     * a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
     * projections imagePoints and the projected (using REF: projectPoints ) objectPoints. The use of RANSAC
     * makes the function resistant to outliers.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePNPRansac for object detection can be found at
     *         opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
     *   </li>
     *   <li>
     *       The default method used to estimate the camera pose for the Minimal Sample Sets step
     *        is #SOLVEPNP_EPNP. Exceptions are:
     *   <ul>
     *     <li>
     *           if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
     *     </li>
     *     <li>
     *           if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
     *     </li>
     *   </ul>
     *   <li>
     *       The method used to estimate the camera pose using all the inliers is defined by the
     *        flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
     *        the method #SOLVEPNP_EPNP will be used instead.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError, double confidence, Mat inliers) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnPRansac_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers.nativeObj);
    }

    /**
     * Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * @param useExtrinsicGuess Parameter used for REF: SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param iterationsCount Number of iterations.
     * @param reprojectionError Inlier threshold value used by the RANSAC procedure. The parameter value
     * is the maximum allowed distance between the observed and computed point projections to consider it
     * an inlier.
     * @param confidence The probability that the algorithm produces a useful result.
     *
     * The function estimates an object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients. This function finds such
     * a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
     * projections imagePoints and the projected (using REF: projectPoints ) objectPoints. The use of RANSAC
     * makes the function resistant to outliers.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePNPRansac for object detection can be found at
     *         opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
     *   </li>
     *   <li>
     *       The default method used to estimate the camera pose for the Minimal Sample Sets step
     *        is #SOLVEPNP_EPNP. Exceptions are:
     *   <ul>
     *     <li>
     *           if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
     *     </li>
     *     <li>
     *           if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
     *     </li>
     *   </ul>
     *   <li>
     *       The method used to estimate the camera pose using all the inliers is defined by the
     *        flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
     *        the method #SOLVEPNP_EPNP will be used instead.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError, double confidence) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnPRansac_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess, iterationsCount, reprojectionError, confidence);
    }

    /**
     * Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * @param useExtrinsicGuess Parameter used for REF: SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param iterationsCount Number of iterations.
     * @param reprojectionError Inlier threshold value used by the RANSAC procedure. The parameter value
     * is the maximum allowed distance between the observed and computed point projections to consider it
     * an inlier.
     *
     * The function estimates an object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients. This function finds such
     * a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
     * projections imagePoints and the projected (using REF: projectPoints ) objectPoints. The use of RANSAC
     * makes the function resistant to outliers.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePNPRansac for object detection can be found at
     *         opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
     *   </li>
     *   <li>
     *       The default method used to estimate the camera pose for the Minimal Sample Sets step
     *        is #SOLVEPNP_EPNP. Exceptions are:
     *   <ul>
     *     <li>
     *           if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
     *     </li>
     *     <li>
     *           if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
     *     </li>
     *   </ul>
     *   <li>
     *       The method used to estimate the camera pose using all the inliers is defined by the
     *        flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
     *        the method #SOLVEPNP_EPNP will be used instead.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnPRansac_3(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess, iterationsCount, reprojectionError);
    }

    /**
     * Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * @param useExtrinsicGuess Parameter used for REF: SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param iterationsCount Number of iterations.
     * is the maximum allowed distance between the observed and computed point projections to consider it
     * an inlier.
     *
     * The function estimates an object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients. This function finds such
     * a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
     * projections imagePoints and the projected (using REF: projectPoints ) objectPoints. The use of RANSAC
     * makes the function resistant to outliers.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePNPRansac for object detection can be found at
     *         opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
     *   </li>
     *   <li>
     *       The default method used to estimate the camera pose for the Minimal Sample Sets step
     *        is #SOLVEPNP_EPNP. Exceptions are:
     *   <ul>
     *     <li>
     *           if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
     *     </li>
     *     <li>
     *           if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
     *     </li>
     *   </ul>
     *   <li>
     *       The method used to estimate the camera pose using all the inliers is defined by the
     *        flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
     *        the method #SOLVEPNP_EPNP will be used instead.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess, int iterationsCount) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnPRansac_4(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess, iterationsCount);
    }

    /**
     * Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * @param useExtrinsicGuess Parameter used for REF: SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * is the maximum allowed distance between the observed and computed point projections to consider it
     * an inlier.
     *
     * The function estimates an object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients. This function finds such
     * a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
     * projections imagePoints and the projected (using REF: projectPoints ) objectPoints. The use of RANSAC
     * makes the function resistant to outliers.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePNPRansac for object detection can be found at
     *         opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
     *   </li>
     *   <li>
     *       The default method used to estimate the camera pose for the Minimal Sample Sets step
     *        is #SOLVEPNP_EPNP. Exceptions are:
     *   <ul>
     *     <li>
     *           if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
     *     </li>
     *     <li>
     *           if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
     *     </li>
     *   </ul>
     *   <li>
     *       The method used to estimate the camera pose using all the inliers is defined by the
     *        flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
     *        the method #SOLVEPNP_EPNP will be used instead.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnPRansac_5(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess);
    }

    /**
     * Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Output translation vector.
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * is the maximum allowed distance between the observed and computed point projections to consider it
     * an inlier.
     *
     * The function estimates an object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients. This function finds such
     * a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
     * projections imagePoints and the projected (using REF: projectPoints ) objectPoints. The use of RANSAC
     * makes the function resistant to outliers.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePNPRansac for object detection can be found at
     *         opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
     *   </li>
     *   <li>
     *       The default method used to estimate the camera pose for the Minimal Sample Sets step
     *        is #SOLVEPNP_EPNP. Exceptions are:
     *   <ul>
     *     <li>
     *           if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
     *     </li>
     *     <li>
     *           if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
     *     </li>
     *   </ul>
     *   <li>
     *       The method used to estimate the camera pose using all the inliers is defined by the
     *        flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
     *        the method #SOLVEPNP_EPNP will be used instead.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec) {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        return solvePnPRansac_6(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj);
    }


    //
    // C++:  bool cv::stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, Mat& H1, Mat& H2, double threshold = 5)
    //

    /**
     * Computes a rectification transform for an uncalibrated stereo camera.
     *
     * @param points1 Array of feature points in the first image.
     * @param points2 The corresponding points in the second image. The same formats as in
     * findFundamentalMat are supported.
     * @param F Input fundamental matrix. It can be computed from the same set of point pairs using
     * findFundamentalMat .
     * @param imgSize Size of the image.
     * @param H1 Output rectification homography matrix for the first image.
     * @param H2 Output rectification homography matrix for the second image.
     * @param threshold Optional threshold used to filter out the outliers. If the parameter is greater
     * than zero, all the point pairs that do not comply with the epipolar geometry (that is, the points
     * for which \(|\texttt{points2[i]}^T*\texttt{F}*\texttt{points1[i]}|&gt;\texttt{threshold}\) ) are
     * rejected prior to computing the homographies. Otherwise, all the points are considered inliers.
     *
     * The function computes the rectification transformations without knowing intrinsic parameters of the
     * cameras and their relative position in the space, which explains the suffix "uncalibrated". Another
     * related difference from stereoRectify is that the function outputs not the rectification
     * transformations in the object (3D) space, but the planar perspective transformations encoded by the
     * homography matrices H1 and H2 . The function implements the algorithm CITE: Hartley99 .
     *
     * <b>Note:</b>
     *    While the algorithm does not need to know the intrinsic parameters of the cameras, it heavily
     *     depends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion,
     *     it would be better to correct it before computing the fundamental matrix and calling this
     *     function. For example, distortion coefficients can be estimated for each head of stereo camera
     *     separately by using calibrateCamera . Then, the images can be corrected using undistort , or
     *     just the point coordinates can be corrected with undistortPoints .
     * @return automatically generated
     */
    public static boolean stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, Mat H1, Mat H2, double threshold) {
        return stereoRectifyUncalibrated_0(points1.nativeObj, points2.nativeObj, F.nativeObj, imgSize.width, imgSize.height, H1.nativeObj, H2.nativeObj, threshold);
    }

    /**
     * Computes a rectification transform for an uncalibrated stereo camera.
     *
     * @param points1 Array of feature points in the first image.
     * @param points2 The corresponding points in the second image. The same formats as in
     * findFundamentalMat are supported.
     * @param F Input fundamental matrix. It can be computed from the same set of point pairs using
     * findFundamentalMat .
     * @param imgSize Size of the image.
     * @param H1 Output rectification homography matrix for the first image.
     * @param H2 Output rectification homography matrix for the second image.
     * than zero, all the point pairs that do not comply with the epipolar geometry (that is, the points
     * for which \(|\texttt{points2[i]}^T*\texttt{F}*\texttt{points1[i]}|&gt;\texttt{threshold}\) ) are
     * rejected prior to computing the homographies. Otherwise, all the points are considered inliers.
     *
     * The function computes the rectification transformations without knowing intrinsic parameters of the
     * cameras and their relative position in the space, which explains the suffix "uncalibrated". Another
     * related difference from stereoRectify is that the function outputs not the rectification
     * transformations in the object (3D) space, but the planar perspective transformations encoded by the
     * homography matrices H1 and H2 . The function implements the algorithm CITE: Hartley99 .
     *
     * <b>Note:</b>
     *    While the algorithm does not need to know the intrinsic parameters of the cameras, it heavily
     *     depends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion,
     *     it would be better to correct it before computing the fundamental matrix and calling this
     *     function. For example, distortion coefficients can be estimated for each head of stereo camera
     *     separately by using calibrateCamera . Then, the images can be corrected using undistort , or
     *     just the point coordinates can be corrected with undistortPoints .
     * @return automatically generated
     */
    public static boolean stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, Mat H1, Mat H2) {
        return stereoRectifyUncalibrated_1(points1.nativeObj, points2.nativeObj, F.nativeObj, imgSize.width, imgSize.height, H1.nativeObj, H2.nativeObj);
    }


    //
    // C++:  double cv::calibrateCamera(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, Mat& stdDeviationsIntrinsics, Mat& stdDeviationsExtrinsics, Mat& perViewErrors, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    //

    /**
     * Finds the camera intrinsic and extrinsic parameters from several views of a calibration
     * pattern.
     *
     * @param objectPoints In the new interface it is a vector of vectors of calibration pattern points in
     * the calibration pattern coordinate space (e.g. std::vector&lt;std::vector&lt;cv::Vec3f&gt;&gt;). The outer
     * vector contains as many elements as the number of pattern views. If the same calibration pattern
     * is shown in each view and it is fully visible, all the vectors will be the same. Although, it is
     * possible to use partially occluded patterns or even different patterns in different views. Then,
     * the vectors will be different. Although the points are 3D, they all lie in the calibration pattern's
     * XY coordinate plane (thus 0 in the Z-coordinate), if the used calibration pattern is a planar rig.
     * In the old interface all the vectors of object points from different views are concatenated
     * together.
     * @param imagePoints In the new interface it is a vector of vectors of the projections of calibration
     * pattern points (e.g. std::vector&lt;std::vector&lt;cv::Vec2f&gt;&gt;). imagePoints.size() and
     * objectPoints.size(), and imagePoints[i].size() and objectPoints[i].size() for each i, must be equal,
     * respectively. In the old interface all the vectors of object points from different views are
     * concatenated together.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
     * @param cameraMatrix Input/output 3x3 floating-point camera matrix
     * \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
     * and/or CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
     * initialized before calling the function.
     * @param distCoeffs Input/output vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements.
     * @param rvecs Output vector of rotation vectors (REF: Rodrigues ) estimated for each pattern view
     * (e.g. std::vector&lt;cv::Mat&gt;&gt;). That is, each i-th rotation vector together with the corresponding
     * i-th translation vector (see the next output parameter description) brings the calibration pattern
     * from the object coordinate space (in which object points are specified) to the camera coordinate
     * space. In more technical terms, the tuple of the i-th rotation and translation vector performs
     * a change of basis from object coordinate space to camera coordinate space. Due to its duality, this
     * tuple is equivalent to the position of the calibration pattern with respect to the camera coordinate
     * space.
     * @param tvecs Output vector of translation vectors estimated for each pattern view, see parameter
     * describtion above.
     * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic
     * parameters. Order of deviations values:
     * \((f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
     *  s_4, \tau_x, \tau_y)\) If one of parameters is not estimated, it's deviation is equals to zero.
     * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic
     * parameters. Order of deviations values: \((R_0, T_0, \dotsc , R_{M - 1}, T_{M - 1})\) where M is
     * the number of pattern views. \(R_i, T_i\) are concatenated 1x3 vectors.
     *  @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * @param flags Different flags that may be zero or a combination of the following values:
     * <ul>
     *   <li>
     *    <b>CALIB_USE_INTRINSIC_GUESS</b> cameraMatrix contains valid initial values of
     * fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     * center ( imageSize is used), and focal distances are computed in a least-squares fashion.
     * Note, that if intrinsic parameters are known, there is no need to use this function just to
     * estimate extrinsic parameters. Use solvePnP instead.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_PRINCIPAL_POINT</b> The principal point is not changed during the global
     * optimization. It stays at the center or at a different location specified when
     * CALIB_USE_INTRINSIC_GUESS is set too.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_ASPECT_RATIO</b> The functions consider only fy as a free parameter. The
     * ratio fx/fy stays the same as in the input cameraMatrix . When
     * CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are
     * ignored, only their ratio is computed and used further.
     *   </li>
     *   <li>
     *    <b>CALIB_ZERO_TANGENT_DIST</b> Tangential distortion coefficients \((p_1, p_2)\) are set
     * to zeros and stay zero.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_K1,...,CALIB_FIX_K6</b> The corresponding radial distortion
     * coefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is
     * set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_RATIONAL_MODEL</b> Coefficients k4, k5, and k6 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the rational model and return 8 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_THIN_PRISM_MODEL</b> Coefficients s1, s2, s3 and s4 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the thin prism model and return 12 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_S1_S2_S3_S4</b> The thin prism distortion coefficients are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_TILTED_MODEL</b> Coefficients tauX and tauY are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_TAUX_TAUY</b> The coefficients of the tilted sensor model are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     * @param criteria Termination criteria for the iterative optimization algorithm.
     *   </li>
     * </ul>
     *
     * @return the overall RMS re-projection error.
     *
     * The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
     * views. The algorithm is based on CITE: Zhang2000 and CITE: BouguetMCT . The coordinates of 3D object
     * points and their corresponding 2D projections in each view must be specified. That may be achieved
     * by using an object with known geometry and easily detectable feature points. Such an object is
     * called a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as
     * a calibration rig (see REF: findChessboardCorners). Currently, initialization of intrinsic
     * parameters (when CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration
     * patterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also
     * be used as long as initial cameraMatrix is provided.
     *
     * The algorithm performs the following steps:
     *
     * <ul>
     *   <li>
     *    Compute the initial intrinsic parameters (the option only available for planar calibration
     *     patterns) or read them from the input parameters. The distortion coefficients are all set to
     *     zeros initially unless some of CALIB_FIX_K? are specified.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *    Estimate the initial camera pose as if the intrinsic parameters have been already known. This is
     *     done using solvePnP .
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *    Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,
     *     that is, the total sum of squared distances between the observed feature points imagePoints and
     *     the projected (using the current estimates for camera parameters and the poses) object points
     *     objectPoints. See projectPoints for details.
     *   </li>
     * </ul>
     *
     * <b>Note:</b>
     *     If you use a non-square (i.e. non-N-by-N) grid and REF: findChessboardCorners for calibration,
     *     and REF: calibrateCamera returns bad values (zero distortion coefficients, \(c_x\) and
     *     \(c_y\) very far from the image center, and/or large differences between \(f_x\) and
     *     \(f_y\) (ratios of 10:1 or more)), then you are probably using patternSize=cvSize(rows,cols)
     *     instead of using patternSize=cvSize(cols,rows) in REF: findChessboardCorners.
     *
     * SEE:
     *    calibrateCameraRO, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate,
     *    undistort
     */
    public static double calibrateCameraExtended(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat stdDeviationsIntrinsics, Mat stdDeviationsExtrinsics, Mat perViewErrors, int flags, TermCriteria criteria) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraExtended_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, stdDeviationsIntrinsics.nativeObj, stdDeviationsExtrinsics.nativeObj, perViewErrors.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds the camera intrinsic and extrinsic parameters from several views of a calibration
     * pattern.
     *
     * @param objectPoints In the new interface it is a vector of vectors of calibration pattern points in
     * the calibration pattern coordinate space (e.g. std::vector&lt;std::vector&lt;cv::Vec3f&gt;&gt;). The outer
     * vector contains as many elements as the number of pattern views. If the same calibration pattern
     * is shown in each view and it is fully visible, all the vectors will be the same. Although, it is
     * possible to use partially occluded patterns or even different patterns in different views. Then,
     * the vectors will be different. Although the points are 3D, they all lie in the calibration pattern's
     * XY coordinate plane (thus 0 in the Z-coordinate), if the used calibration pattern is a planar rig.
     * In the old interface all the vectors of object points from different views are concatenated
     * together.
     * @param imagePoints In the new interface it is a vector of vectors of the projections of calibration
     * pattern points (e.g. std::vector&lt;std::vector&lt;cv::Vec2f&gt;&gt;). imagePoints.size() and
     * objectPoints.size(), and imagePoints[i].size() and objectPoints[i].size() for each i, must be equal,
     * respectively. In the old interface all the vectors of object points from different views are
     * concatenated together.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
     * @param cameraMatrix Input/output 3x3 floating-point camera matrix
     * \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
     * and/or CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
     * initialized before calling the function.
     * @param distCoeffs Input/output vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements.
     * @param rvecs Output vector of rotation vectors (REF: Rodrigues ) estimated for each pattern view
     * (e.g. std::vector&lt;cv::Mat&gt;&gt;). That is, each i-th rotation vector together with the corresponding
     * i-th translation vector (see the next output parameter description) brings the calibration pattern
     * from the object coordinate space (in which object points are specified) to the camera coordinate
     * space. In more technical terms, the tuple of the i-th rotation and translation vector performs
     * a change of basis from object coordinate space to camera coordinate space. Due to its duality, this
     * tuple is equivalent to the position of the calibration pattern with respect to the camera coordinate
     * space.
     * @param tvecs Output vector of translation vectors estimated for each pattern view, see parameter
     * describtion above.
     * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic
     * parameters. Order of deviations values:
     * \((f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
     *  s_4, \tau_x, \tau_y)\) If one of parameters is not estimated, it's deviation is equals to zero.
     * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic
     * parameters. Order of deviations values: \((R_0, T_0, \dotsc , R_{M - 1}, T_{M - 1})\) where M is
     * the number of pattern views. \(R_i, T_i\) are concatenated 1x3 vectors.
     *  @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * @param flags Different flags that may be zero or a combination of the following values:
     * <ul>
     *   <li>
     *    <b>CALIB_USE_INTRINSIC_GUESS</b> cameraMatrix contains valid initial values of
     * fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     * center ( imageSize is used), and focal distances are computed in a least-squares fashion.
     * Note, that if intrinsic parameters are known, there is no need to use this function just to
     * estimate extrinsic parameters. Use solvePnP instead.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_PRINCIPAL_POINT</b> The principal point is not changed during the global
     * optimization. It stays at the center or at a different location specified when
     * CALIB_USE_INTRINSIC_GUESS is set too.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_ASPECT_RATIO</b> The functions consider only fy as a free parameter. The
     * ratio fx/fy stays the same as in the input cameraMatrix . When
     * CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are
     * ignored, only their ratio is computed and used further.
     *   </li>
     *   <li>
     *    <b>CALIB_ZERO_TANGENT_DIST</b> Tangential distortion coefficients \((p_1, p_2)\) are set
     * to zeros and stay zero.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_K1,...,CALIB_FIX_K6</b> The corresponding radial distortion
     * coefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is
     * set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_RATIONAL_MODEL</b> Coefficients k4, k5, and k6 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the rational model and return 8 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_THIN_PRISM_MODEL</b> Coefficients s1, s2, s3 and s4 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the thin prism model and return 12 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_S1_S2_S3_S4</b> The thin prism distortion coefficients are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_TILTED_MODEL</b> Coefficients tauX and tauY are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_TAUX_TAUY</b> The coefficients of the tilted sensor model are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     * </ul>
     *
     * @return the overall RMS re-projection error.
     *
     * The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
     * views. The algorithm is based on CITE: Zhang2000 and CITE: BouguetMCT . The coordinates of 3D object
     * points and their corresponding 2D projections in each view must be specified. That may be achieved
     * by using an object with known geometry and easily detectable feature points. Such an object is
     * called a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as
     * a calibration rig (see REF: findChessboardCorners). Currently, initialization of intrinsic
     * parameters (when CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration
     * patterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also
     * be used as long as initial cameraMatrix is provided.
     *
     * The algorithm performs the following steps:
     *
     * <ul>
     *   <li>
     *    Compute the initial intrinsic parameters (the option only available for planar calibration
     *     patterns) or read them from the input parameters. The distortion coefficients are all set to
     *     zeros initially unless some of CALIB_FIX_K? are specified.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *    Estimate the initial camera pose as if the intrinsic parameters have been already known. This is
     *     done using solvePnP .
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *    Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,
     *     that is, the total sum of squared distances between the observed feature points imagePoints and
     *     the projected (using the current estimates for camera parameters and the poses) object points
     *     objectPoints. See projectPoints for details.
     *   </li>
     * </ul>
     *
     * <b>Note:</b>
     *     If you use a non-square (i.e. non-N-by-N) grid and REF: findChessboardCorners for calibration,
     *     and REF: calibrateCamera returns bad values (zero distortion coefficients, \(c_x\) and
     *     \(c_y\) very far from the image center, and/or large differences between \(f_x\) and
     *     \(f_y\) (ratios of 10:1 or more)), then you are probably using patternSize=cvSize(rows,cols)
     *     instead of using patternSize=cvSize(cols,rows) in REF: findChessboardCorners.
     *
     * SEE:
     *    calibrateCameraRO, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate,
     *    undistort
     */
    public static double calibrateCameraExtended(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat stdDeviationsIntrinsics, Mat stdDeviationsExtrinsics, Mat perViewErrors, int flags) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraExtended_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, stdDeviationsIntrinsics.nativeObj, stdDeviationsExtrinsics.nativeObj, perViewErrors.nativeObj, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds the camera intrinsic and extrinsic parameters from several views of a calibration
     * pattern.
     *
     * @param objectPoints In the new interface it is a vector of vectors of calibration pattern points in
     * the calibration pattern coordinate space (e.g. std::vector&lt;std::vector&lt;cv::Vec3f&gt;&gt;). The outer
     * vector contains as many elements as the number of pattern views. If the same calibration pattern
     * is shown in each view and it is fully visible, all the vectors will be the same. Although, it is
     * possible to use partially occluded patterns or even different patterns in different views. Then,
     * the vectors will be different. Although the points are 3D, they all lie in the calibration pattern's
     * XY coordinate plane (thus 0 in the Z-coordinate), if the used calibration pattern is a planar rig.
     * In the old interface all the vectors of object points from different views are concatenated
     * together.
     * @param imagePoints In the new interface it is a vector of vectors of the projections of calibration
     * pattern points (e.g. std::vector&lt;std::vector&lt;cv::Vec2f&gt;&gt;). imagePoints.size() and
     * objectPoints.size(), and imagePoints[i].size() and objectPoints[i].size() for each i, must be equal,
     * respectively. In the old interface all the vectors of object points from different views are
     * concatenated together.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
     * @param cameraMatrix Input/output 3x3 floating-point camera matrix
     * \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
     * and/or CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
     * initialized before calling the function.
     * @param distCoeffs Input/output vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements.
     * @param rvecs Output vector of rotation vectors (REF: Rodrigues ) estimated for each pattern view
     * (e.g. std::vector&lt;cv::Mat&gt;&gt;). That is, each i-th rotation vector together with the corresponding
     * i-th translation vector (see the next output parameter description) brings the calibration pattern
     * from the object coordinate space (in which object points are specified) to the camera coordinate
     * space. In more technical terms, the tuple of the i-th rotation and translation vector performs
     * a change of basis from object coordinate space to camera coordinate space. Due to its duality, this
     * tuple is equivalent to the position of the calibration pattern with respect to the camera coordinate
     * space.
     * @param tvecs Output vector of translation vectors estimated for each pattern view, see parameter
     * describtion above.
     * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic
     * parameters. Order of deviations values:
     * \((f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
     *  s_4, \tau_x, \tau_y)\) If one of parameters is not estimated, it's deviation is equals to zero.
     * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic
     * parameters. Order of deviations values: \((R_0, T_0, \dotsc , R_{M - 1}, T_{M - 1})\) where M is
     * the number of pattern views. \(R_i, T_i\) are concatenated 1x3 vectors.
     *  @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * <ul>
     *   <li>
     *    <b>CALIB_USE_INTRINSIC_GUESS</b> cameraMatrix contains valid initial values of
     * fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     * center ( imageSize is used), and focal distances are computed in a least-squares fashion.
     * Note, that if intrinsic parameters are known, there is no need to use this function just to
     * estimate extrinsic parameters. Use solvePnP instead.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_PRINCIPAL_POINT</b> The principal point is not changed during the global
     * optimization. It stays at the center or at a different location specified when
     * CALIB_USE_INTRINSIC_GUESS is set too.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_ASPECT_RATIO</b> The functions consider only fy as a free parameter. The
     * ratio fx/fy stays the same as in the input cameraMatrix . When
     * CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are
     * ignored, only their ratio is computed and used further.
     *   </li>
     *   <li>
     *    <b>CALIB_ZERO_TANGENT_DIST</b> Tangential distortion coefficients \((p_1, p_2)\) are set
     * to zeros and stay zero.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_K1,...,CALIB_FIX_K6</b> The corresponding radial distortion
     * coefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is
     * set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_RATIONAL_MODEL</b> Coefficients k4, k5, and k6 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the rational model and return 8 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_THIN_PRISM_MODEL</b> Coefficients s1, s2, s3 and s4 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the thin prism model and return 12 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_S1_S2_S3_S4</b> The thin prism distortion coefficients are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_TILTED_MODEL</b> Coefficients tauX and tauY are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_TAUX_TAUY</b> The coefficients of the tilted sensor model are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     * </ul>
     *
     * @return the overall RMS re-projection error.
     *
     * The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
     * views. The algorithm is based on CITE: Zhang2000 and CITE: BouguetMCT . The coordinates of 3D object
     * points and their corresponding 2D projections in each view must be specified. That may be achieved
     * by using an object with known geometry and easily detectable feature points. Such an object is
     * called a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as
     * a calibration rig (see REF: findChessboardCorners). Currently, initialization of intrinsic
     * parameters (when CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration
     * patterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also
     * be used as long as initial cameraMatrix is provided.
     *
     * The algorithm performs the following steps:
     *
     * <ul>
     *   <li>
     *    Compute the initial intrinsic parameters (the option only available for planar calibration
     *     patterns) or read them from the input parameters. The distortion coefficients are all set to
     *     zeros initially unless some of CALIB_FIX_K? are specified.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *    Estimate the initial camera pose as if the intrinsic parameters have been already known. This is
     *     done using solvePnP .
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *    Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,
     *     that is, the total sum of squared distances between the observed feature points imagePoints and
     *     the projected (using the current estimates for camera parameters and the poses) object points
     *     objectPoints. See projectPoints for details.
     *   </li>
     * </ul>
     *
     * <b>Note:</b>
     *     If you use a non-square (i.e. non-N-by-N) grid and REF: findChessboardCorners for calibration,
     *     and REF: calibrateCamera returns bad values (zero distortion coefficients, \(c_x\) and
     *     \(c_y\) very far from the image center, and/or large differences between \(f_x\) and
     *     \(f_y\) (ratios of 10:1 or more)), then you are probably using patternSize=cvSize(rows,cols)
     *     instead of using patternSize=cvSize(cols,rows) in REF: findChessboardCorners.
     *
     * SEE:
     *    calibrateCameraRO, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate,
     *    undistort
     */
    public static double calibrateCameraExtended(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat stdDeviationsIntrinsics, Mat stdDeviationsExtrinsics, Mat perViewErrors) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraExtended_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, stdDeviationsIntrinsics.nativeObj, stdDeviationsExtrinsics.nativeObj, perViewErrors.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  double cv::calibrateCamera(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    //

    public static double calibrateCamera(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, int flags, TermCriteria criteria) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCamera_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    public static double calibrateCamera(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, int flags) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCamera_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    public static double calibrateCamera(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCamera_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  double cv::calibrateCameraRO(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, int iFixedPoint, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, Mat& newObjPoints, Mat& stdDeviationsIntrinsics, Mat& stdDeviationsExtrinsics, Mat& stdDeviationsObjPoints, Mat& perViewErrors, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    //

    /**
     * Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
     *
     * This function is an extension of calibrateCamera() with the method of releasing object which was
     * proposed in CITE: strobl2011iccv. In many common cases with inaccurate, unmeasured, roughly planar
     * targets (calibration plates), this method can dramatically improve the precision of the estimated
     * camera parameters. Both the object-releasing method and standard method are supported by this
     * function. Use the parameter <b>iFixedPoint</b> for method selection. In the internal implementation,
     * calibrateCamera() is a wrapper for this function.
     *
     * @param objectPoints Vector of vectors of calibration pattern points in the calibration pattern
     * coordinate space. See calibrateCamera() for details. If the method of releasing object to be used,
     * the identical calibration board must be used in each view and it must be fully visible, and all
     * objectPoints[i] must be the same and all points should be roughly close to a plane. <b>The calibration
     * target has to be rigid, or at least static if the camera (rather than the calibration target) is
     * shifted for grabbing images.</b>
     * @param imagePoints Vector of vectors of the projections of calibration pattern points. See
     * calibrateCamera() for details.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
     * @param iFixedPoint The index of the 3D object point in objectPoints[0] to be fixed. It also acts as
     * a switch for calibration method selection. If object-releasing method to be used, pass in the
     * parameter in the range of [1, objectPoints[0].size()-2], otherwise a value out of this range will
     * make standard calibration method selected. Usually the top-right corner point of the calibration
     * board grid is recommended to be fixed when object-releasing method being utilized. According to
     * \cite strobl2011iccv, two other points are also fixed. In this implementation, objectPoints[0].front
     * and objectPoints[0].back.z are used. With object-releasing method, accurate rvecs, tvecs and
     * newObjPoints are only possible if coordinates of these three fixed points are accurate enough.
     * @param cameraMatrix Output 3x3 floating-point camera matrix. See calibrateCamera() for details.
     * @param distCoeffs Output vector of distortion coefficients. See calibrateCamera() for details.
     * @param rvecs Output vector of rotation vectors estimated for each pattern view. See calibrateCamera()
     * for details.
     * @param tvecs Output vector of translation vectors estimated for each pattern view.
     * @param newObjPoints The updated output vector of calibration pattern points. The coordinates might
     * be scaled based on three fixed points. The returned coordinates are accurate only if the above
     * mentioned three fixed points are accurate. If not needed, noArray() can be passed in. This parameter
     * is ignored with standard calibration method.
     * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
     * See calibrateCamera() for details.
     * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
     * See calibrateCamera() for details.
     * @param stdDeviationsObjPoints Output vector of standard deviations estimated for refined coordinates
     * of calibration pattern points. It has the same size and order as objectPoints[0] vector. This
     * parameter is ignored with standard calibration method.
     *  @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * @param flags Different flags that may be zero or a combination of some predefined values. See
     * calibrateCamera() for details. If the method of releasing object is used, the calibration time may
     * be much longer. CALIB_USE_QR or CALIB_USE_LU could be used for faster calibration with potentially
     * less precise and less stable in some rare cases.
     * @param criteria Termination criteria for the iterative optimization algorithm.
     *
     * @return the overall RMS re-projection error.
     *
     * The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
     * views. The algorithm is based on CITE: Zhang2000, CITE: BouguetMCT and CITE: strobl2011iccv. See
     * calibrateCamera() for other detailed explanations.
     * SEE:
     *    calibrateCamera, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort
     */
    public static double calibrateCameraROExtended(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, int iFixedPoint, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat newObjPoints, Mat stdDeviationsIntrinsics, Mat stdDeviationsExtrinsics, Mat stdDeviationsObjPoints, Mat perViewErrors, int flags, TermCriteria criteria) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraROExtended_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, iFixedPoint, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, newObjPoints.nativeObj, stdDeviationsIntrinsics.nativeObj, stdDeviationsExtrinsics.nativeObj, stdDeviationsObjPoints.nativeObj, perViewErrors.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
     *
     * This function is an extension of calibrateCamera() with the method of releasing object which was
     * proposed in CITE: strobl2011iccv. In many common cases with inaccurate, unmeasured, roughly planar
     * targets (calibration plates), this method can dramatically improve the precision of the estimated
     * camera parameters. Both the object-releasing method and standard method are supported by this
     * function. Use the parameter <b>iFixedPoint</b> for method selection. In the internal implementation,
     * calibrateCamera() is a wrapper for this function.
     *
     * @param objectPoints Vector of vectors of calibration pattern points in the calibration pattern
     * coordinate space. See calibrateCamera() for details. If the method of releasing object to be used,
     * the identical calibration board must be used in each view and it must be fully visible, and all
     * objectPoints[i] must be the same and all points should be roughly close to a plane. <b>The calibration
     * target has to be rigid, or at least static if the camera (rather than the calibration target) is
     * shifted for grabbing images.</b>
     * @param imagePoints Vector of vectors of the projections of calibration pattern points. See
     * calibrateCamera() for details.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
     * @param iFixedPoint The index of the 3D object point in objectPoints[0] to be fixed. It also acts as
     * a switch for calibration method selection. If object-releasing method to be used, pass in the
     * parameter in the range of [1, objectPoints[0].size()-2], otherwise a value out of this range will
     * make standard calibration method selected. Usually the top-right corner point of the calibration
     * board grid is recommended to be fixed when object-releasing method being utilized. According to
     * \cite strobl2011iccv, two other points are also fixed. In this implementation, objectPoints[0].front
     * and objectPoints[0].back.z are used. With object-releasing method, accurate rvecs, tvecs and
     * newObjPoints are only possible if coordinates of these three fixed points are accurate enough.
     * @param cameraMatrix Output 3x3 floating-point camera matrix. See calibrateCamera() for details.
     * @param distCoeffs Output vector of distortion coefficients. See calibrateCamera() for details.
     * @param rvecs Output vector of rotation vectors estimated for each pattern view. See calibrateCamera()
     * for details.
     * @param tvecs Output vector of translation vectors estimated for each pattern view.
     * @param newObjPoints The updated output vector of calibration pattern points. The coordinates might
     * be scaled based on three fixed points. The returned coordinates are accurate only if the above
     * mentioned three fixed points are accurate. If not needed, noArray() can be passed in. This parameter
     * is ignored with standard calibration method.
     * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
     * See calibrateCamera() for details.
     * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
     * See calibrateCamera() for details.
     * @param stdDeviationsObjPoints Output vector of standard deviations estimated for refined coordinates
     * of calibration pattern points. It has the same size and order as objectPoints[0] vector. This
     * parameter is ignored with standard calibration method.
     *  @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * @param flags Different flags that may be zero or a combination of some predefined values. See
     * calibrateCamera() for details. If the method of releasing object is used, the calibration time may
     * be much longer. CALIB_USE_QR or CALIB_USE_LU could be used for faster calibration with potentially
     * less precise and less stable in some rare cases.
     *
     * @return the overall RMS re-projection error.
     *
     * The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
     * views. The algorithm is based on CITE: Zhang2000, CITE: BouguetMCT and CITE: strobl2011iccv. See
     * calibrateCamera() for other detailed explanations.
     * SEE:
     *    calibrateCamera, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort
     */
    public static double calibrateCameraROExtended(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, int iFixedPoint, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat newObjPoints, Mat stdDeviationsIntrinsics, Mat stdDeviationsExtrinsics, Mat stdDeviationsObjPoints, Mat perViewErrors, int flags) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraROExtended_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, iFixedPoint, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, newObjPoints.nativeObj, stdDeviationsIntrinsics.nativeObj, stdDeviationsExtrinsics.nativeObj, stdDeviationsObjPoints.nativeObj, perViewErrors.nativeObj, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
     *
     * This function is an extension of calibrateCamera() with the method of releasing object which was
     * proposed in CITE: strobl2011iccv. In many common cases with inaccurate, unmeasured, roughly planar
     * targets (calibration plates), this method can dramatically improve the precision of the estimated
     * camera parameters. Both the object-releasing method and standard method are supported by this
     * function. Use the parameter <b>iFixedPoint</b> for method selection. In the internal implementation,
     * calibrateCamera() is a wrapper for this function.
     *
     * @param objectPoints Vector of vectors of calibration pattern points in the calibration pattern
     * coordinate space. See calibrateCamera() for details. If the method of releasing object to be used,
     * the identical calibration board must be used in each view and it must be fully visible, and all
     * objectPoints[i] must be the same and all points should be roughly close to a plane. <b>The calibration
     * target has to be rigid, or at least static if the camera (rather than the calibration target) is
     * shifted for grabbing images.</b>
     * @param imagePoints Vector of vectors of the projections of calibration pattern points. See
     * calibrateCamera() for details.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
     * @param iFixedPoint The index of the 3D object point in objectPoints[0] to be fixed. It also acts as
     * a switch for calibration method selection. If object-releasing method to be used, pass in the
     * parameter in the range of [1, objectPoints[0].size()-2], otherwise a value out of this range will
     * make standard calibration method selected. Usually the top-right corner point of the calibration
     * board grid is recommended to be fixed when object-releasing method being utilized. According to
     * \cite strobl2011iccv, two other points are also fixed. In this implementation, objectPoints[0].front
     * and objectPoints[0].back.z are used. With object-releasing method, accurate rvecs, tvecs and
     * newObjPoints are only possible if coordinates of these three fixed points are accurate enough.
     * @param cameraMatrix Output 3x3 floating-point camera matrix. See calibrateCamera() for details.
     * @param distCoeffs Output vector of distortion coefficients. See calibrateCamera() for details.
     * @param rvecs Output vector of rotation vectors estimated for each pattern view. See calibrateCamera()
     * for details.
     * @param tvecs Output vector of translation vectors estimated for each pattern view.
     * @param newObjPoints The updated output vector of calibration pattern points. The coordinates might
     * be scaled based on three fixed points. The returned coordinates are accurate only if the above
     * mentioned three fixed points are accurate. If not needed, noArray() can be passed in. This parameter
     * is ignored with standard calibration method.
     * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
     * See calibrateCamera() for details.
     * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
     * See calibrateCamera() for details.
     * @param stdDeviationsObjPoints Output vector of standard deviations estimated for refined coordinates
     * of calibration pattern points. It has the same size and order as objectPoints[0] vector. This
     * parameter is ignored with standard calibration method.
     *  @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * calibrateCamera() for details. If the method of releasing object is used, the calibration time may
     * be much longer. CALIB_USE_QR or CALIB_USE_LU could be used for faster calibration with potentially
     * less precise and less stable in some rare cases.
     *
     * @return the overall RMS re-projection error.
     *
     * The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
     * views. The algorithm is based on CITE: Zhang2000, CITE: BouguetMCT and CITE: strobl2011iccv. See
     * calibrateCamera() for other detailed explanations.
     * SEE:
     *    calibrateCamera, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort
     */
    public static double calibrateCameraROExtended(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, int iFixedPoint, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat newObjPoints, Mat stdDeviationsIntrinsics, Mat stdDeviationsExtrinsics, Mat stdDeviationsObjPoints, Mat perViewErrors) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraROExtended_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, iFixedPoint, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, newObjPoints.nativeObj, stdDeviationsIntrinsics.nativeObj, stdDeviationsExtrinsics.nativeObj, stdDeviationsObjPoints.nativeObj, perViewErrors.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  double cv::calibrateCameraRO(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, int iFixedPoint, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, Mat& newObjPoints, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    //

    public static double calibrateCameraRO(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, int iFixedPoint, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat newObjPoints, int flags, TermCriteria criteria) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraRO_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, iFixedPoint, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, newObjPoints.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    public static double calibrateCameraRO(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, int iFixedPoint, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat newObjPoints, int flags) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraRO_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, iFixedPoint, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, newObjPoints.nativeObj, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    public static double calibrateCameraRO(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, int iFixedPoint, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, Mat newObjPoints) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCameraRO_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, iFixedPoint, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, newObjPoints.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  double cv::sampsonDistance(Mat pt1, Mat pt2, Mat F)
    //

    /**
     * Calculates the Sampson Distance between two points.
     *
     * The function cv::sampsonDistance calculates and returns the first order approximation of the geometric error as:
     * \(
     * sd( \texttt{pt1} , \texttt{pt2} )=
     * \frac{(\texttt{pt2}^t \cdot \texttt{F} \cdot \texttt{pt1})^2}
     * {((\texttt{F} \cdot \texttt{pt1})(0))^2 +
     * ((\texttt{F} \cdot \texttt{pt1})(1))^2 +
     * ((\texttt{F}^t \cdot \texttt{pt2})(0))^2 +
     * ((\texttt{F}^t \cdot \texttt{pt2})(1))^2}
     * \)
     * The fundamental matrix may be calculated using the cv::findFundamentalMat function. See CITE: HartleyZ00 11.4.3 for details.
     * @param pt1 first homogeneous 2d point
     * @param pt2 second homogeneous 2d point
     * @param F fundamental matrix
     * @return The computed Sampson distance.
     */
    public static double sampsonDistance(Mat pt1, Mat pt2, Mat F) {
        return sampsonDistance_0(pt1.nativeObj, pt2.nativeObj, F.nativeObj);
    }


    //
    // C++:  double cv::stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size imageSize, Mat& R, Mat& T, Mat& E, Mat& F, Mat& perViewErrors, int flags = CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))
    //

    /**
     * Calibrates a stereo camera set up. This function finds the intrinsic parameters
     * for each of the two cameras and the extrinsic parameters between the two cameras.
     *
     * @param objectPoints Vector of vectors of the calibration pattern points. The same structure as
     * in REF: calibrateCamera. For each pattern view, both cameras need to see the same object
     * points. Therefore, objectPoints.size(), imagePoints1.size(), and imagePoints2.size() need to be
     * equal as well as objectPoints[i].size(), imagePoints1[i].size(), and imagePoints2[i].size() need to
     * be equal for each i.
     * @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
     * observed by the first camera. The same structure as in REF: calibrateCamera.
     * @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
     * observed by the second camera. The same structure as in REF: calibrateCamera.
     * @param cameraMatrix1 Input/output camera matrix for the first camera, the same as in
     * REF: calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below.
     * @param distCoeffs1 Input/output vector of distortion coefficients, the same as in
     * REF: calibrateCamera.
     * @param cameraMatrix2 Input/output second camera matrix for the second camera. See description for
     * cameraMatrix1.
     * @param distCoeffs2 Input/output lens distortion coefficients for the second camera. See
     * description for distCoeffs1.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrices.
     * @param R Output rotation matrix. Together with the translation vector T, this matrix brings
     * points given in the first camera's coordinate system to points in the second camera's
     * coordinate system. In more technical terms, the tuple of R and T performs a change of basis
     * from the first camera's coordinate system to the second camera's coordinate system. Due to its
     * duality, this tuple is equivalent to the position of the first camera with respect to the
     * second camera coordinate system.
     * @param T Output translation vector, see description above.
     * @param E Output essential matrix.
     * @param F Output fundamental matrix.
     * @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * @param flags Different flags that may be zero or a combination of the following values:
     * <ul>
     *   <li>
     *    <b>CALIB_FIX_INTRINSIC</b> Fix cameraMatrix? and distCoeffs? so that only R, T, E, and F
     * matrices are estimated.
     *   </li>
     *   <li>
     *    <b>CALIB_USE_INTRINSIC_GUESS</b> Optimize some or all of the intrinsic parameters
     * according to the specified flags. Initial values are provided by the user.
     *   </li>
     *   <li>
     *    <b>CALIB_USE_EXTRINSIC_GUESS</b> R and T contain valid initial values that are optimized further.
     * Otherwise R and T are initialized to the median value of the pattern views (each dimension separately).
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_PRINCIPAL_POINT</b> Fix the principal points during the optimization.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_FOCAL_LENGTH</b> Fix \(f^{(j)}_x\) and \(f^{(j)}_y\) .
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_ASPECT_RATIO</b> Optimize \(f^{(j)}_y\) . Fix the ratio \(f^{(j)}_x/f^{(j)}_y\)
     * .
     *   </li>
     *   <li>
     *    <b>CALIB_SAME_FOCAL_LENGTH</b> Enforce \(f^{(0)}_x=f^{(1)}_x\) and \(f^{(0)}_y=f^{(1)}_y\) .
     *   </li>
     *   <li>
     *    <b>CALIB_ZERO_TANGENT_DIST</b> Set tangential distortion coefficients for each camera to
     * zeros and fix there.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_K1,...,CALIB_FIX_K6</b> Do not change the corresponding radial
     * distortion coefficient during the optimization. If CALIB_USE_INTRINSIC_GUESS is set,
     * the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_RATIONAL_MODEL</b> Enable coefficients k4, k5, and k6. To provide the backward
     * compatibility, this extra flag should be explicitly specified to make the calibration
     * function use the rational model and return 8 coefficients. If the flag is not set, the
     * function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_THIN_PRISM_MODEL</b> Coefficients s1, s2, s3 and s4 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the thin prism model and return 12 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_S1_S2_S3_S4</b> The thin prism distortion coefficients are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_TILTED_MODEL</b> Coefficients tauX and tauY are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_TAUX_TAUY</b> The coefficients of the tilted sensor model are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     * @param criteria Termination criteria for the iterative optimization algorithm.
     *   </li>
     * </ul>
     *
     * The function estimates the transformation between two cameras making a stereo pair. If one computes
     * the poses of an object relative to the first camera and to the second camera,
     * ( \(R_1\),\(T_1\) ) and (\(R_2\),\(T_2\)), respectively, for a stereo camera where the
     * relative position and orientation between the two cameras are fixed, then those poses definitely
     * relate to each other. This means, if the relative position and orientation (\(R\),\(T\)) of the
     * two cameras is known, it is possible to compute (\(R_2\),\(T_2\)) when (\(R_1\),\(T_1\)) is
     * given. This is what the described function does. It computes (\(R\),\(T\)) such that:
     *
     * \(R_2=R R_1\)
     * \(T_2=R T_1 + T.\)
     *
     * Therefore, one can compute the coordinate representation of a 3D point for the second camera's
     * coordinate system when given the point's coordinate representation in the first camera's coordinate
     * system:
     *
     * \(\begin{bmatrix}
     * X_2 \\
     * Y_2 \\
     * Z_2 \\
     * 1
     * \end{bmatrix} = \begin{bmatrix}
     * R &amp; T \\
     * 0 &amp; 1
     * \end{bmatrix} \begin{bmatrix}
     * X_1 \\
     * Y_1 \\
     * Z_1 \\
     * 1
     * \end{bmatrix}.\)
     *
     *
     * Optionally, it computes the essential matrix E:
     *
     * \(E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} R\)
     *
     * where \(T_i\) are components of the translation vector \(T\) : \(T=[T_0, T_1, T_2]^T\) .
     * And the function can also compute the fundamental matrix F:
     *
     * \(F = cameraMatrix2^{-T} E cameraMatrix1^{-1}\)
     *
     * Besides the stereo-related information, the function can also perform a full calibration of each of
     * the two cameras. However, due to the high dimensionality of the parameter space and noise in the
     * input data, the function can diverge from the correct solution. If the intrinsic parameters can be
     * estimated with high accuracy for each of the cameras individually (for example, using
     * calibrateCamera ), you are recommended to do so and then pass CALIB_FIX_INTRINSIC flag to the
     * function along with the computed intrinsic parameters. Otherwise, if all the parameters are
     * estimated at once, it makes sense to restrict some parameters, for example, pass
     * CALIB_SAME_FOCAL_LENGTH and CALIB_ZERO_TANGENT_DIST flags, which is usually a
     * reasonable assumption.
     *
     * Similarly to calibrateCamera, the function minimizes the total re-projection error for all the
     * points in all the available views from both cameras. The function returns the final value of the
     * re-projection error.
     * @return automatically generated
     */
    public static double stereoCalibrateExtended(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F, Mat perViewErrors, int flags, TermCriteria criteria) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return stereoCalibrateExtended_0(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj, perViewErrors.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
    }

    /**
     * Calibrates a stereo camera set up. This function finds the intrinsic parameters
     * for each of the two cameras and the extrinsic parameters between the two cameras.
     *
     * @param objectPoints Vector of vectors of the calibration pattern points. The same structure as
     * in REF: calibrateCamera. For each pattern view, both cameras need to see the same object
     * points. Therefore, objectPoints.size(), imagePoints1.size(), and imagePoints2.size() need to be
     * equal as well as objectPoints[i].size(), imagePoints1[i].size(), and imagePoints2[i].size() need to
     * be equal for each i.
     * @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
     * observed by the first camera. The same structure as in REF: calibrateCamera.
     * @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
     * observed by the second camera. The same structure as in REF: calibrateCamera.
     * @param cameraMatrix1 Input/output camera matrix for the first camera, the same as in
     * REF: calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below.
     * @param distCoeffs1 Input/output vector of distortion coefficients, the same as in
     * REF: calibrateCamera.
     * @param cameraMatrix2 Input/output second camera matrix for the second camera. See description for
     * cameraMatrix1.
     * @param distCoeffs2 Input/output lens distortion coefficients for the second camera. See
     * description for distCoeffs1.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrices.
     * @param R Output rotation matrix. Together with the translation vector T, this matrix brings
     * points given in the first camera's coordinate system to points in the second camera's
     * coordinate system. In more technical terms, the tuple of R and T performs a change of basis
     * from the first camera's coordinate system to the second camera's coordinate system. Due to its
     * duality, this tuple is equivalent to the position of the first camera with respect to the
     * second camera coordinate system.
     * @param T Output translation vector, see description above.
     * @param E Output essential matrix.
     * @param F Output fundamental matrix.
     * @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * @param flags Different flags that may be zero or a combination of the following values:
     * <ul>
     *   <li>
     *    <b>CALIB_FIX_INTRINSIC</b> Fix cameraMatrix? and distCoeffs? so that only R, T, E, and F
     * matrices are estimated.
     *   </li>
     *   <li>
     *    <b>CALIB_USE_INTRINSIC_GUESS</b> Optimize some or all of the intrinsic parameters
     * according to the specified flags. Initial values are provided by the user.
     *   </li>
     *   <li>
     *    <b>CALIB_USE_EXTRINSIC_GUESS</b> R and T contain valid initial values that are optimized further.
     * Otherwise R and T are initialized to the median value of the pattern views (each dimension separately).
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_PRINCIPAL_POINT</b> Fix the principal points during the optimization.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_FOCAL_LENGTH</b> Fix \(f^{(j)}_x\) and \(f^{(j)}_y\) .
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_ASPECT_RATIO</b> Optimize \(f^{(j)}_y\) . Fix the ratio \(f^{(j)}_x/f^{(j)}_y\)
     * .
     *   </li>
     *   <li>
     *    <b>CALIB_SAME_FOCAL_LENGTH</b> Enforce \(f^{(0)}_x=f^{(1)}_x\) and \(f^{(0)}_y=f^{(1)}_y\) .
     *   </li>
     *   <li>
     *    <b>CALIB_ZERO_TANGENT_DIST</b> Set tangential distortion coefficients for each camera to
     * zeros and fix there.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_K1,...,CALIB_FIX_K6</b> Do not change the corresponding radial
     * distortion coefficient during the optimization. If CALIB_USE_INTRINSIC_GUESS is set,
     * the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_RATIONAL_MODEL</b> Enable coefficients k4, k5, and k6. To provide the backward
     * compatibility, this extra flag should be explicitly specified to make the calibration
     * function use the rational model and return 8 coefficients. If the flag is not set, the
     * function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_THIN_PRISM_MODEL</b> Coefficients s1, s2, s3 and s4 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the thin prism model and return 12 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_S1_S2_S3_S4</b> The thin prism distortion coefficients are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_TILTED_MODEL</b> Coefficients tauX and tauY are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_TAUX_TAUY</b> The coefficients of the tilted sensor model are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     * </ul>
     *
     * The function estimates the transformation between two cameras making a stereo pair. If one computes
     * the poses of an object relative to the first camera and to the second camera,
     * ( \(R_1\),\(T_1\) ) and (\(R_2\),\(T_2\)), respectively, for a stereo camera where the
     * relative position and orientation between the two cameras are fixed, then those poses definitely
     * relate to each other. This means, if the relative position and orientation (\(R\),\(T\)) of the
     * two cameras is known, it is possible to compute (\(R_2\),\(T_2\)) when (\(R_1\),\(T_1\)) is
     * given. This is what the described function does. It computes (\(R\),\(T\)) such that:
     *
     * \(R_2=R R_1\)
     * \(T_2=R T_1 + T.\)
     *
     * Therefore, one can compute the coordinate representation of a 3D point for the second camera's
     * coordinate system when given the point's coordinate representation in the first camera's coordinate
     * system:
     *
     * \(\begin{bmatrix}
     * X_2 \\
     * Y_2 \\
     * Z_2 \\
     * 1
     * \end{bmatrix} = \begin{bmatrix}
     * R &amp; T \\
     * 0 &amp; 1
     * \end{bmatrix} \begin{bmatrix}
     * X_1 \\
     * Y_1 \\
     * Z_1 \\
     * 1
     * \end{bmatrix}.\)
     *
     *
     * Optionally, it computes the essential matrix E:
     *
     * \(E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} R\)
     *
     * where \(T_i\) are components of the translation vector \(T\) : \(T=[T_0, T_1, T_2]^T\) .
     * And the function can also compute the fundamental matrix F:
     *
     * \(F = cameraMatrix2^{-T} E cameraMatrix1^{-1}\)
     *
     * Besides the stereo-related information, the function can also perform a full calibration of each of
     * the two cameras. However, due to the high dimensionality of the parameter space and noise in the
     * input data, the function can diverge from the correct solution. If the intrinsic parameters can be
     * estimated with high accuracy for each of the cameras individually (for example, using
     * calibrateCamera ), you are recommended to do so and then pass CALIB_FIX_INTRINSIC flag to the
     * function along with the computed intrinsic parameters. Otherwise, if all the parameters are
     * estimated at once, it makes sense to restrict some parameters, for example, pass
     * CALIB_SAME_FOCAL_LENGTH and CALIB_ZERO_TANGENT_DIST flags, which is usually a
     * reasonable assumption.
     *
     * Similarly to calibrateCamera, the function minimizes the total re-projection error for all the
     * points in all the available views from both cameras. The function returns the final value of the
     * re-projection error.
     * @return automatically generated
     */
    public static double stereoCalibrateExtended(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F, Mat perViewErrors, int flags) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return stereoCalibrateExtended_1(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj, perViewErrors.nativeObj, flags);
    }

    /**
     * Calibrates a stereo camera set up. This function finds the intrinsic parameters
     * for each of the two cameras and the extrinsic parameters between the two cameras.
     *
     * @param objectPoints Vector of vectors of the calibration pattern points. The same structure as
     * in REF: calibrateCamera. For each pattern view, both cameras need to see the same object
     * points. Therefore, objectPoints.size(), imagePoints1.size(), and imagePoints2.size() need to be
     * equal as well as objectPoints[i].size(), imagePoints1[i].size(), and imagePoints2[i].size() need to
     * be equal for each i.
     * @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
     * observed by the first camera. The same structure as in REF: calibrateCamera.
     * @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
     * observed by the second camera. The same structure as in REF: calibrateCamera.
     * @param cameraMatrix1 Input/output camera matrix for the first camera, the same as in
     * REF: calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below.
     * @param distCoeffs1 Input/output vector of distortion coefficients, the same as in
     * REF: calibrateCamera.
     * @param cameraMatrix2 Input/output second camera matrix for the second camera. See description for
     * cameraMatrix1.
     * @param distCoeffs2 Input/output lens distortion coefficients for the second camera. See
     * description for distCoeffs1.
     * @param imageSize Size of the image used only to initialize the intrinsic camera matrices.
     * @param R Output rotation matrix. Together with the translation vector T, this matrix brings
     * points given in the first camera's coordinate system to points in the second camera's
     * coordinate system. In more technical terms, the tuple of R and T performs a change of basis
     * from the first camera's coordinate system to the second camera's coordinate system. Due to its
     * duality, this tuple is equivalent to the position of the first camera with respect to the
     * second camera coordinate system.
     * @param T Output translation vector, see description above.
     * @param E Output essential matrix.
     * @param F Output fundamental matrix.
     * @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
     * <ul>
     *   <li>
     *    <b>CALIB_FIX_INTRINSIC</b> Fix cameraMatrix? and distCoeffs? so that only R, T, E, and F
     * matrices are estimated.
     *   </li>
     *   <li>
     *    <b>CALIB_USE_INTRINSIC_GUESS</b> Optimize some or all of the intrinsic parameters
     * according to the specified flags. Initial values are provided by the user.
     *   </li>
     *   <li>
     *    <b>CALIB_USE_EXTRINSIC_GUESS</b> R and T contain valid initial values that are optimized further.
     * Otherwise R and T are initialized to the median value of the pattern views (each dimension separately).
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_PRINCIPAL_POINT</b> Fix the principal points during the optimization.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_FOCAL_LENGTH</b> Fix \(f^{(j)}_x\) and \(f^{(j)}_y\) .
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_ASPECT_RATIO</b> Optimize \(f^{(j)}_y\) . Fix the ratio \(f^{(j)}_x/f^{(j)}_y\)
     * .
     *   </li>
     *   <li>
     *    <b>CALIB_SAME_FOCAL_LENGTH</b> Enforce \(f^{(0)}_x=f^{(1)}_x\) and \(f^{(0)}_y=f^{(1)}_y\) .
     *   </li>
     *   <li>
     *    <b>CALIB_ZERO_TANGENT_DIST</b> Set tangential distortion coefficients for each camera to
     * zeros and fix there.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_K1,...,CALIB_FIX_K6</b> Do not change the corresponding radial
     * distortion coefficient during the optimization. If CALIB_USE_INTRINSIC_GUESS is set,
     * the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_RATIONAL_MODEL</b> Enable coefficients k4, k5, and k6. To provide the backward
     * compatibility, this extra flag should be explicitly specified to make the calibration
     * function use the rational model and return 8 coefficients. If the flag is not set, the
     * function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_THIN_PRISM_MODEL</b> Coefficients s1, s2, s3 and s4 are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the thin prism model and return 12 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_S1_S2_S3_S4</b> The thin prism distortion coefficients are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     *   <li>
     *    <b>CALIB_TILTED_MODEL</b> Coefficients tauX and tauY are enabled. To provide the
     * backward compatibility, this extra flag should be explicitly specified to make the
     * calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
     * set, the function computes and returns only 5 distortion coefficients.
     *   </li>
     *   <li>
     *    <b>CALIB_FIX_TAUX_TAUY</b> The coefficients of the tilted sensor model are not changed during
     * the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
     * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
     *   </li>
     * </ul>
     *
     * The function estimates the transformation between two cameras making a stereo pair. If one computes
     * the poses of an object relative to the first camera and to the second camera,
     * ( \(R_1\),\(T_1\) ) and (\(R_2\),\(T_2\)), respectively, for a stereo camera where the
     * relative position and orientation between the two cameras are fixed, then those poses definitely
     * relate to each other. This means, if the relative position and orientation (\(R\),\(T\)) of the
     * two cameras is known, it is possible to compute (\(R_2\),\(T_2\)) when (\(R_1\),\(T_1\)) is
     * given. This is what the described function does. It computes (\(R\),\(T\)) such that:
     *
     * \(R_2=R R_1\)
     * \(T_2=R T_1 + T.\)
     *
     * Therefore, one can compute the coordinate representation of a 3D point for the second camera's
     * coordinate system when given the point's coordinate representation in the first camera's coordinate
     * system:
     *
     * \(\begin{bmatrix}
     * X_2 \\
     * Y_2 \\
     * Z_2 \\
     * 1
     * \end{bmatrix} = \begin{bmatrix}
     * R &amp; T \\
     * 0 &amp; 1
     * \end{bmatrix} \begin{bmatrix}
     * X_1 \\
     * Y_1 \\
     * Z_1 \\
     * 1
     * \end{bmatrix}.\)
     *
     *
     * Optionally, it computes the essential matrix E:
     *
     * \(E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} R\)
     *
     * where \(T_i\) are components of the translation vector \(T\) : \(T=[T_0, T_1, T_2]^T\) .
     * And the function can also compute the fundamental matrix F:
     *
     * \(F = cameraMatrix2^{-T} E cameraMatrix1^{-1}\)
     *
     * Besides the stereo-related information, the function can also perform a full calibration of each of
     * the two cameras. However, due to the high dimensionality of the parameter space and noise in the
     * input data, the function can diverge from the correct solution. If the intrinsic parameters can be
     * estimated with high accuracy for each of the cameras individually (for example, using
     * calibrateCamera ), you are recommended to do so and then pass CALIB_FIX_INTRINSIC flag to the
     * function along with the computed intrinsic parameters. Otherwise, if all the parameters are
     * estimated at once, it makes sense to restrict some parameters, for example, pass
     * CALIB_SAME_FOCAL_LENGTH and CALIB_ZERO_TANGENT_DIST flags, which is usually a
     * reasonable assumption.
     *
     * Similarly to calibrateCamera, the function minimizes the total re-projection error for all the
     * points in all the available views from both cameras. The function returns the final value of the
     * re-projection error.
     * @return automatically generated
     */
    public static double stereoCalibrateExtended(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F, Mat perViewErrors) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return stereoCalibrateExtended_2(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj, perViewErrors.nativeObj);
    }


    //
    // C++:  double cv::stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size imageSize, Mat& R, Mat& T, Mat& E, Mat& F, int flags = CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))
    //

    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F, int flags, TermCriteria criteria) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return stereoCalibrate_0(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
    }

    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F, int flags) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return stereoCalibrate_1(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj, flags);
    }

    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return stereoCalibrate_2(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj);
    }


    //
    // C++:  double cv::fisheye::calibrate(vector_Mat objectPoints, vector_Mat imagePoints, Size image_size, Mat& K, Mat& D, vector_Mat& rvecs, vector_Mat& tvecs, int flags = 0, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON))
    //

    /**
     * Performs camera calibaration
     *
     *     @param objectPoints vector of vectors of calibration pattern points in the calibration pattern
     *     coordinate space.
     *     @param imagePoints vector of vectors of the projections of calibration pattern points.
     *     imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to
     *     objectPoints[i].size() for each i.
     *     @param image_size Size of the image used only to initialize the intrinsic camera matrix.
     *     @param K Output 3x3 floating-point camera matrix
     *     \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) . If
     *     fisheye::CALIB_USE_INTRINSIC_GUESS/ is specified, some or all of fx, fy, cx, cy must be
     *     initialized before calling the function.
     *     @param D Output vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view.
     *     That is, each k-th rotation vector together with the corresponding k-th translation vector (see
     *     the next output parameter description) brings the calibration pattern from the model coordinate
     *     space (in which object points are specified) to the world coordinate space, that is, a real
     *     position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
     *     @param tvecs Output vector of translation vectors estimated for each pattern view.
     *     @param flags Different flags that may be zero or a combination of the following values:
     * <ul>
     *   <li>
     *        <b>fisheye::CALIB_USE_INTRINSIC_GUESS</b> cameraMatrix contains valid initial values of
     *     fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     *     center ( imageSize is used), and focal distances are computed in a least-squares fashion.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_RECOMPUTE_EXTRINSIC</b> Extrinsic will be recomputed after each iteration
     *     of intrinsic optimization.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_CHECK_COND</b> The functions will check validity of condition number.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_SKEW</b> Skew coefficient (alpha) is set to zero and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_K1..fisheye::CALIB_FIX_K4</b> Selected distortion coefficients
     *     are set to zeros and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_PRINCIPAL_POINT</b> The principal point is not changed during the global
     * optimization. It stays at the center or at a different location specified when CALIB_USE_INTRINSIC_GUESS is set too.
     *     @param criteria Termination criteria for the iterative optimization algorithm.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static double fisheye_calibrate(List<Mat> objectPoints, List<Mat> imagePoints, Size image_size, Mat K, Mat D, List<Mat> rvecs, List<Mat> tvecs, int flags, TermCriteria criteria) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = fisheye_calibrate_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, image_size.width, image_size.height, K.nativeObj, D.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Performs camera calibaration
     *
     *     @param objectPoints vector of vectors of calibration pattern points in the calibration pattern
     *     coordinate space.
     *     @param imagePoints vector of vectors of the projections of calibration pattern points.
     *     imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to
     *     objectPoints[i].size() for each i.
     *     @param image_size Size of the image used only to initialize the intrinsic camera matrix.
     *     @param K Output 3x3 floating-point camera matrix
     *     \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) . If
     *     fisheye::CALIB_USE_INTRINSIC_GUESS/ is specified, some or all of fx, fy, cx, cy must be
     *     initialized before calling the function.
     *     @param D Output vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view.
     *     That is, each k-th rotation vector together with the corresponding k-th translation vector (see
     *     the next output parameter description) brings the calibration pattern from the model coordinate
     *     space (in which object points are specified) to the world coordinate space, that is, a real
     *     position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
     *     @param tvecs Output vector of translation vectors estimated for each pattern view.
     *     @param flags Different flags that may be zero or a combination of the following values:
     * <ul>
     *   <li>
     *        <b>fisheye::CALIB_USE_INTRINSIC_GUESS</b> cameraMatrix contains valid initial values of
     *     fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     *     center ( imageSize is used), and focal distances are computed in a least-squares fashion.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_RECOMPUTE_EXTRINSIC</b> Extrinsic will be recomputed after each iteration
     *     of intrinsic optimization.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_CHECK_COND</b> The functions will check validity of condition number.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_SKEW</b> Skew coefficient (alpha) is set to zero and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_K1..fisheye::CALIB_FIX_K4</b> Selected distortion coefficients
     *     are set to zeros and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_PRINCIPAL_POINT</b> The principal point is not changed during the global
     * optimization. It stays at the center or at a different location specified when CALIB_USE_INTRINSIC_GUESS is set too.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static double fisheye_calibrate(List<Mat> objectPoints, List<Mat> imagePoints, Size image_size, Mat K, Mat D, List<Mat> rvecs, List<Mat> tvecs, int flags) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = fisheye_calibrate_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, image_size.width, image_size.height, K.nativeObj, D.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Performs camera calibaration
     *
     *     @param objectPoints vector of vectors of calibration pattern points in the calibration pattern
     *     coordinate space.
     *     @param imagePoints vector of vectors of the projections of calibration pattern points.
     *     imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to
     *     objectPoints[i].size() for each i.
     *     @param image_size Size of the image used only to initialize the intrinsic camera matrix.
     *     @param K Output 3x3 floating-point camera matrix
     *     \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) . If
     *     fisheye::CALIB_USE_INTRINSIC_GUESS/ is specified, some or all of fx, fy, cx, cy must be
     *     initialized before calling the function.
     *     @param D Output vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view.
     *     That is, each k-th rotation vector together with the corresponding k-th translation vector (see
     *     the next output parameter description) brings the calibration pattern from the model coordinate
     *     space (in which object points are specified) to the world coordinate space, that is, a real
     *     position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
     *     @param tvecs Output vector of translation vectors estimated for each pattern view.
     * <ul>
     *   <li>
     *        <b>fisheye::CALIB_USE_INTRINSIC_GUESS</b> cameraMatrix contains valid initial values of
     *     fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     *     center ( imageSize is used), and focal distances are computed in a least-squares fashion.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_RECOMPUTE_EXTRINSIC</b> Extrinsic will be recomputed after each iteration
     *     of intrinsic optimization.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_CHECK_COND</b> The functions will check validity of condition number.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_SKEW</b> Skew coefficient (alpha) is set to zero and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_K1..fisheye::CALIB_FIX_K4</b> Selected distortion coefficients
     *     are set to zeros and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_PRINCIPAL_POINT</b> The principal point is not changed during the global
     * optimization. It stays at the center or at a different location specified when CALIB_USE_INTRINSIC_GUESS is set too.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static double fisheye_calibrate(List<Mat> objectPoints, List<Mat> imagePoints, Size image_size, Mat K, Mat D, List<Mat> rvecs, List<Mat> tvecs) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = fisheye_calibrate_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, image_size.width, image_size.height, K.nativeObj, D.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  double cv::fisheye::stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& K1, Mat& D1, Mat& K2, Mat& D2, Size imageSize, Mat& R, Mat& T, int flags = fisheye::CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON))
    //

    /**
     * Performs stereo calibration
     *
     *     @param objectPoints Vector of vectors of the calibration pattern points.
     *     @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
     *     observed by the first camera.
     *     @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
     *     observed by the second camera.
     *     @param K1 Input/output first camera matrix:
     *     \(\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\) , \(j = 0,\, 1\) . If
     *     any of fisheye::CALIB_USE_INTRINSIC_GUESS , fisheye::CALIB_FIX_INTRINSIC are specified,
     *     some or all of the matrix components must be initialized.
     *     @param D1 Input/output vector of distortion coefficients \((k_1, k_2, k_3, k_4)\) of 4 elements.
     *     @param K2 Input/output second camera matrix. The parameter is similar to K1 .
     *     @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
     *     similar to D1 .
     *     @param imageSize Size of the image used only to initialize intrinsic camera matrix.
     *     @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
     *     @param T Output translation vector between the coordinate systems of the cameras.
     *     @param flags Different flags that may be zero or a combination of the following values:
     * <ul>
     *   <li>
     *        <b>fisheye::CALIB_FIX_INTRINSIC</b> Fix K1, K2? and D1, D2? so that only R, T matrices
     *     are estimated.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_USE_INTRINSIC_GUESS</b> K1, K2 contains valid initial values of
     *     fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     *     center (imageSize is used), and focal distances are computed in a least-squares fashion.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_RECOMPUTE_EXTRINSIC</b> Extrinsic will be recomputed after each iteration
     *     of intrinsic optimization.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_CHECK_COND</b> The functions will check validity of condition number.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_SKEW</b> Skew coefficient (alpha) is set to zero and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_K1..4</b> Selected distortion coefficients are set to zeros and stay
     *     zero.
     *     @param criteria Termination criteria for the iterative optimization algorithm.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static double fisheye_stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat T, int flags, TermCriteria criteria) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return fisheye_stereoCalibrate_0(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
    }

    /**
     * Performs stereo calibration
     *
     *     @param objectPoints Vector of vectors of the calibration pattern points.
     *     @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
     *     observed by the first camera.
     *     @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
     *     observed by the second camera.
     *     @param K1 Input/output first camera matrix:
     *     \(\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\) , \(j = 0,\, 1\) . If
     *     any of fisheye::CALIB_USE_INTRINSIC_GUESS , fisheye::CALIB_FIX_INTRINSIC are specified,
     *     some or all of the matrix components must be initialized.
     *     @param D1 Input/output vector of distortion coefficients \((k_1, k_2, k_3, k_4)\) of 4 elements.
     *     @param K2 Input/output second camera matrix. The parameter is similar to K1 .
     *     @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
     *     similar to D1 .
     *     @param imageSize Size of the image used only to initialize intrinsic camera matrix.
     *     @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
     *     @param T Output translation vector between the coordinate systems of the cameras.
     *     @param flags Different flags that may be zero or a combination of the following values:
     * <ul>
     *   <li>
     *        <b>fisheye::CALIB_FIX_INTRINSIC</b> Fix K1, K2? and D1, D2? so that only R, T matrices
     *     are estimated.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_USE_INTRINSIC_GUESS</b> K1, K2 contains valid initial values of
     *     fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     *     center (imageSize is used), and focal distances are computed in a least-squares fashion.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_RECOMPUTE_EXTRINSIC</b> Extrinsic will be recomputed after each iteration
     *     of intrinsic optimization.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_CHECK_COND</b> The functions will check validity of condition number.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_SKEW</b> Skew coefficient (alpha) is set to zero and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_K1..4</b> Selected distortion coefficients are set to zeros and stay
     *     zero.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static double fisheye_stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat T, int flags) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return fisheye_stereoCalibrate_1(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, flags);
    }

    /**
     * Performs stereo calibration
     *
     *     @param objectPoints Vector of vectors of the calibration pattern points.
     *     @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
     *     observed by the first camera.
     *     @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
     *     observed by the second camera.
     *     @param K1 Input/output first camera matrix:
     *     \(\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\) , \(j = 0,\, 1\) . If
     *     any of fisheye::CALIB_USE_INTRINSIC_GUESS , fisheye::CALIB_FIX_INTRINSIC are specified,
     *     some or all of the matrix components must be initialized.
     *     @param D1 Input/output vector of distortion coefficients \((k_1, k_2, k_3, k_4)\) of 4 elements.
     *     @param K2 Input/output second camera matrix. The parameter is similar to K1 .
     *     @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
     *     similar to D1 .
     *     @param imageSize Size of the image used only to initialize intrinsic camera matrix.
     *     @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
     *     @param T Output translation vector between the coordinate systems of the cameras.
     * <ul>
     *   <li>
     *        <b>fisheye::CALIB_FIX_INTRINSIC</b> Fix K1, K2? and D1, D2? so that only R, T matrices
     *     are estimated.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_USE_INTRINSIC_GUESS</b> K1, K2 contains valid initial values of
     *     fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
     *     center (imageSize is used), and focal distances are computed in a least-squares fashion.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_RECOMPUTE_EXTRINSIC</b> Extrinsic will be recomputed after each iteration
     *     of intrinsic optimization.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_CHECK_COND</b> The functions will check validity of condition number.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_SKEW</b> Skew coefficient (alpha) is set to zero and stay zero.
     *   </li>
     *   <li>
     *        <b>fisheye::CALIB_FIX_K1..4</b> Selected distortion coefficients are set to zeros and stay
     *     zero.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static double fisheye_stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat T) {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        return fisheye_stereoCalibrate_2(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj);
    }


    //
    // C++:  float cv::rectify3Collinear(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Mat cameraMatrix3, Mat distCoeffs3, vector_Mat imgpt1, vector_Mat imgpt3, Size imageSize, Mat R12, Mat T12, Mat R13, Mat T13, Mat& R1, Mat& R2, Mat& R3, Mat& P1, Mat& P2, Mat& P3, Mat& Q, double alpha, Size newImgSize, Rect* roi1, Rect* roi2, int flags)
    //

    public static float rectify3Collinear(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Mat cameraMatrix3, Mat distCoeffs3, List<Mat> imgpt1, List<Mat> imgpt3, Size imageSize, Mat R12, Mat T12, Mat R13, Mat T13, Mat R1, Mat R2, Mat R3, Mat P1, Mat P2, Mat P3, Mat Q, double alpha, Size newImgSize, Rect roi1, Rect roi2, int flags) {
        Mat imgpt1_mat = Converters.vector_Mat_to_Mat(imgpt1);
        Mat imgpt3_mat = Converters.vector_Mat_to_Mat(imgpt3);
        double[] roi1_out = new double[4];
        double[] roi2_out = new double[4];
        float retVal = rectify3Collinear_0(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, cameraMatrix3.nativeObj, distCoeffs3.nativeObj, imgpt1_mat.nativeObj, imgpt3_mat.nativeObj, imageSize.width, imageSize.height, R12.nativeObj, T12.nativeObj, R13.nativeObj, T13.nativeObj, R1.nativeObj, R2.nativeObj, R3.nativeObj, P1.nativeObj, P2.nativeObj, P3.nativeObj, Q.nativeObj, alpha, newImgSize.width, newImgSize.height, roi1_out, roi2_out, flags);
        if(roi1!=null){ roi1.x = (int)roi1_out[0]; roi1.y = (int)roi1_out[1]; roi1.width = (int)roi1_out[2]; roi1.height = (int)roi1_out[3]; } 
        if(roi2!=null){ roi2.x = (int)roi2_out[0]; roi2.y = (int)roi2_out[1]; roi2.width = (int)roi2_out[2]; roi2.height = (int)roi2_out[3]; } 
        return retVal;
    }


    //
    // C++:  int cv::decomposeHomographyMat(Mat H, Mat K, vector_Mat& rotations, vector_Mat& translations, vector_Mat& normals)
    //

    /**
     * Decompose a homography matrix to rotation(s), translation(s) and plane normal(s).
     *
     * @param H The input homography matrix between two images.
     * @param K The input intrinsic camera calibration matrix.
     * @param rotations Array of rotation matrices.
     * @param translations Array of translation matrices.
     * @param normals Array of plane normal matrices.
     *
     * This function extracts relative camera motion between two views of a planar object and returns up to
     * four mathematical solution tuples of rotation, translation, and plane normal. The decomposition of
     * the homography matrix H is described in detail in CITE: Malis.
     *
     * If the homography H, induced by the plane, gives the constraint
     * \(s_i \vecthree{x'_i}{y'_i}{1} \sim H \vecthree{x_i}{y_i}{1}\) on the source image points
     * \(p_i\) and the destination image points \(p'_i\), then the tuple of rotations[k] and
     * translations[k] is a change of basis from the source camera's coordinate system to the destination
     * camera's coordinate system. However, by decomposing H, one can only get the translation normalized
     * by the (typically unknown) depth of the scene, i.e. its direction but with normalized length.
     *
     * If point correspondences are available, at least two solutions may further be invalidated, by
     * applying positive depth constraint, i.e. all points must be in front of the camera.
     * @return automatically generated
     */
    public static int decomposeHomographyMat(Mat H, Mat K, List<Mat> rotations, List<Mat> translations, List<Mat> normals) {
        Mat rotations_mat = new Mat();
        Mat translations_mat = new Mat();
        Mat normals_mat = new Mat();
        int retVal = decomposeHomographyMat_0(H.nativeObj, K.nativeObj, rotations_mat.nativeObj, translations_mat.nativeObj, normals_mat.nativeObj);
        Converters.Mat_to_vector_Mat(rotations_mat, rotations);
        rotations_mat.release();
        Converters.Mat_to_vector_Mat(translations_mat, translations);
        translations_mat.release();
        Converters.Mat_to_vector_Mat(normals_mat, normals);
        normals_mat.release();
        return retVal;
    }


    //
    // C++:  int cv::estimateAffine3D(Mat src, Mat dst, Mat& out, Mat& inliers, double ransacThreshold = 3, double confidence = 0.99)
    //

    /**
     * Computes an optimal affine transformation between two 3D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * z\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; a_{13}\\
     * a_{21} &amp; a_{22} &amp; a_{23}\\
     * a_{31} &amp; a_{32} &amp; a_{33}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * Z\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * b_3\\
     * \end{bmatrix}
     * \)
     *
     * @param src First input 3D point set containing \((X,Y,Z)\).
     * @param dst Second input 3D point set containing \((x,y,z)\).
     * @param out Output 3D affine transformation matrix \(3 \times 4\) of the form
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; a_{13} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; a_{23} &amp; b_2\\
     * a_{31} &amp; a_{32} &amp; a_{33} &amp; b_3\\
     * \end{bmatrix}
     * \)
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * @param ransacThreshold Maximum reprojection error in the RANSAC algorithm to consider a point as
     * an inlier.
     * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     *
     * The function estimates an optimal 3D affine transformation between two 3D point sets using the
     * RANSAC algorithm.
     * @return automatically generated
     */
    public static int estimateAffine3D(Mat src, Mat dst, Mat out, Mat inliers, double ransacThreshold, double confidence) {
        return estimateAffine3D_0(src.nativeObj, dst.nativeObj, out.nativeObj, inliers.nativeObj, ransacThreshold, confidence);
    }

    /**
     * Computes an optimal affine transformation between two 3D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * z\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; a_{13}\\
     * a_{21} &amp; a_{22} &amp; a_{23}\\
     * a_{31} &amp; a_{32} &amp; a_{33}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * Z\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * b_3\\
     * \end{bmatrix}
     * \)
     *
     * @param src First input 3D point set containing \((X,Y,Z)\).
     * @param dst Second input 3D point set containing \((x,y,z)\).
     * @param out Output 3D affine transformation matrix \(3 \times 4\) of the form
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; a_{13} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; a_{23} &amp; b_2\\
     * a_{31} &amp; a_{32} &amp; a_{33} &amp; b_3\\
     * \end{bmatrix}
     * \)
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * @param ransacThreshold Maximum reprojection error in the RANSAC algorithm to consider a point as
     * an inlier.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     *
     * The function estimates an optimal 3D affine transformation between two 3D point sets using the
     * RANSAC algorithm.
     * @return automatically generated
     */
    public static int estimateAffine3D(Mat src, Mat dst, Mat out, Mat inliers, double ransacThreshold) {
        return estimateAffine3D_1(src.nativeObj, dst.nativeObj, out.nativeObj, inliers.nativeObj, ransacThreshold);
    }

    /**
     * Computes an optimal affine transformation between two 3D point sets.
     *
     * It computes
     * \(
     * \begin{bmatrix}
     * x\\
     * y\\
     * z\\
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; a_{13}\\
     * a_{21} &amp; a_{22} &amp; a_{23}\\
     * a_{31} &amp; a_{32} &amp; a_{33}\\
     * \end{bmatrix}
     * \begin{bmatrix}
     * X\\
     * Y\\
     * Z\\
     * \end{bmatrix}
     * +
     * \begin{bmatrix}
     * b_1\\
     * b_2\\
     * b_3\\
     * \end{bmatrix}
     * \)
     *
     * @param src First input 3D point set containing \((X,Y,Z)\).
     * @param dst Second input 3D point set containing \((x,y,z)\).
     * @param out Output 3D affine transformation matrix \(3 \times 4\) of the form
     * \(
     * \begin{bmatrix}
     * a_{11} &amp; a_{12} &amp; a_{13} &amp; b_1\\
     * a_{21} &amp; a_{22} &amp; a_{23} &amp; b_2\\
     * a_{31} &amp; a_{32} &amp; a_{33} &amp; b_3\\
     * \end{bmatrix}
     * \)
     * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
     * an inlier.
     * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
     * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
     *
     * The function estimates an optimal 3D affine transformation between two 3D point sets using the
     * RANSAC algorithm.
     * @return automatically generated
     */
    public static int estimateAffine3D(Mat src, Mat dst, Mat out, Mat inliers) {
        return estimateAffine3D_2(src.nativeObj, dst.nativeObj, out.nativeObj, inliers.nativeObj);
    }


    //
    // C++:  int cv::recoverPose(Mat E, Mat points1, Mat points2, Mat& R, Mat& t, double focal = 1.0, Point2d pp = Point2d(0, 0), Mat& mask = Mat())
    //

    /**
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * description below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * @param focal Focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * @param pp principal point of the camera.
     * @param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(A =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat R, Mat t, double focal, Point pp, Mat mask) {
        return recoverPose_0(E.nativeObj, points1.nativeObj, points2.nativeObj, R.nativeObj, t.nativeObj, focal, pp.x, pp.y, mask.nativeObj);
    }

    /**
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * description below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * @param focal Focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * @param pp principal point of the camera.
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(A =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat R, Mat t, double focal, Point pp) {
        return recoverPose_1(E.nativeObj, points1.nativeObj, points2.nativeObj, R.nativeObj, t.nativeObj, focal, pp.x, pp.y);
    }

    /**
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * description below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * @param focal Focal length of the camera. Note that this function assumes that points1 and points2
     * are feature points from cameras with same focal length and principal point.
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(A =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat R, Mat t, double focal) {
        return recoverPose_2(E.nativeObj, points1.nativeObj, points2.nativeObj, R.nativeObj, t.nativeObj, focal);
    }

    /**
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * description below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * are feature points from cameras with same focal length and principal point.
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     *
     * This function differs from the one above that it computes camera matrix from focal length and
     * principal point:
     *
     * \(A =
     * \begin{bmatrix}
     * f &amp; 0 &amp; x_{pp}  \\
     * 0 &amp; f &amp; y_{pp}  \\
     * 0 &amp; 0 &amp; 1
     * \end{bmatrix}\)
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat R, Mat t) {
        return recoverPose_3(E.nativeObj, points1.nativeObj, points2.nativeObj, R.nativeObj, t.nativeObj);
    }


    //
    // C++:  int cv::recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat& R, Mat& t, Mat& mask = Mat())
    //

    /**
     * Recovers the relative camera rotation and the translation from an estimated essential
     * matrix and the corresponding points in two images, using cheirality check. Returns the number of
     * inliers that pass the check.
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param cameraMatrix Camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * described below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * @param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     *
     * This function decomposes an essential matrix using REF: decomposeEssentialMat and then verifies
     * possible pose hypotheses by doing cheirality check. The cheirality check means that the
     * triangulated 3D points should have positive depth. Some details can be found in CITE: Nister03.
     *
     * This function can be used to process the output E and mask from REF: findEssentialMat. In this
     * scenario, points1 and points2 are the same input for findEssentialMat.:
     * <code>
     *     // Example. Estimation of fundamental matrix using the RANSAC algorithm
     *     int point_count = 100;
     *     vector&lt;Point2f&gt; points1(point_count);
     *     vector&lt;Point2f&gt; points2(point_count);
     *
     *     // initialize the points here ...
     *     for( int i = 0; i &lt; point_count; i++ )
     *     {
     *         points1[i] = ...;
     *         points2[i] = ...;
     *     }
     *
     *     // cametra matrix with both focal lengths = 1, and principal point = (0, 0)
     *     Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
     *
     *     Mat E, R, t, mask;
     *
     *     E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
     *     recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
     * </code>
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat R, Mat t, Mat mask) {
        return recoverPose_4(E.nativeObj, points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, R.nativeObj, t.nativeObj, mask.nativeObj);
    }

    /**
     * Recovers the relative camera rotation and the translation from an estimated essential
     * matrix and the corresponding points in two images, using cheirality check. Returns the number of
     * inliers that pass the check.
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1 .
     * @param cameraMatrix Camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * described below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     *
     * This function decomposes an essential matrix using REF: decomposeEssentialMat and then verifies
     * possible pose hypotheses by doing cheirality check. The cheirality check means that the
     * triangulated 3D points should have positive depth. Some details can be found in CITE: Nister03.
     *
     * This function can be used to process the output E and mask from REF: findEssentialMat. In this
     * scenario, points1 and points2 are the same input for findEssentialMat.:
     * <code>
     *     // Example. Estimation of fundamental matrix using the RANSAC algorithm
     *     int point_count = 100;
     *     vector&lt;Point2f&gt; points1(point_count);
     *     vector&lt;Point2f&gt; points2(point_count);
     *
     *     // initialize the points here ...
     *     for( int i = 0; i &lt; point_count; i++ )
     *     {
     *         points1[i] = ...;
     *         points2[i] = ...;
     *     }
     *
     *     // cametra matrix with both focal lengths = 1, and principal point = (0, 0)
     *     Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
     *
     *     Mat E, R, t, mask;
     *
     *     E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
     *     recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
     * </code>
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat R, Mat t) {
        return recoverPose_5(E.nativeObj, points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, R.nativeObj, t.nativeObj);
    }


    //
    // C++:  int cv::recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat& R, Mat& t, double distanceThresh, Mat& mask = Mat(), Mat& triangulatedPoints = Mat())
    //

    /**
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1.
     * @param cameraMatrix Camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * description below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * @param distanceThresh threshold distance which is used to filter out far away points (i.e. infinite
     * points).
     * @param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     * @param triangulatedPoints 3D points which were reconstructed by triangulation.
     *
     * This function differs from the one above that it outputs the triangulated 3D point that are used for
     * the cheirality check.
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat R, Mat t, double distanceThresh, Mat mask, Mat triangulatedPoints) {
        return recoverPose_6(E.nativeObj, points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, R.nativeObj, t.nativeObj, distanceThresh, mask.nativeObj, triangulatedPoints.nativeObj);
    }

    /**
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1.
     * @param cameraMatrix Camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * description below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * @param distanceThresh threshold distance which is used to filter out far away points (i.e. infinite
     * points).
     * @param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     *
     * This function differs from the one above that it outputs the triangulated 3D point that are used for
     * the cheirality check.
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat R, Mat t, double distanceThresh, Mat mask) {
        return recoverPose_7(E.nativeObj, points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, R.nativeObj, t.nativeObj, distanceThresh, mask.nativeObj);
    }

    /**
     *
     * @param E The input essential matrix.
     * @param points1 Array of N 2D points from the first image. The point coordinates should be
     * floating-point (single or double precision).
     * @param points2 Array of the second image points of the same size and format as points1.
     * @param cameraMatrix Camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * Note that this function assumes that points1 and points2 are feature points from cameras with the
     * same camera matrix.
     * @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
     * that performs a change of basis from the first camera's coordinate system to the second camera's
     * coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
     * description below.
     * @param t Output translation vector. This vector is obtained by REF: decomposeEssentialMat and
     * therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
     * length.
     * @param distanceThresh threshold distance which is used to filter out far away points (i.e. infinite
     * points).
     * inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
     * recover pose. In the output mask only inliers which pass the cheirality check.
     *
     * This function differs from the one above that it outputs the triangulated 3D point that are used for
     * the cheirality check.
     * @return automatically generated
     */
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat R, Mat t, double distanceThresh) {
        return recoverPose_8(E.nativeObj, points1.nativeObj, points2.nativeObj, cameraMatrix.nativeObj, R.nativeObj, t.nativeObj, distanceThresh);
    }


    //
    // C++:  int cv::solveP3P(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, int flags)
    //

    /**
     * Finds an object pose from 3 3D-2D point correspondences.
     *
     * @param objectPoints Array of object points in the object coordinate space, 3x3 1-channel or
     * 1x3/3x1 3-channel. vector&lt;Point3f&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, 3x2 1-channel or 1x3/3x1 2-channel.
     *  vector&lt;Point2f&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvecs Output rotation vectors (see REF: Rodrigues ) that, together with tvecs, brings points from
     * the model coordinate system to the camera coordinate system. A P3P problem has up to 4 solutions.
     * @param tvecs Output translation vectors.
     * @param flags Method for solving a P3P problem:
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke and S. Roumeliotis.
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     *   </li>
     * </ul>
     *
     * The function estimates the object pose given 3 object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients.
     *
     * <b>Note:</b>
     * The solutions are sorted by reprojection errors (lowest to highest).
     * @return automatically generated
     */
    public static int solveP3P(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, int flags) {
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        int retVal = solveP3P_0(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  int cv::solvePnPGeneric(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, bool useExtrinsicGuess = false, SolvePnPMethod flags = SOLVEPNP_ITERATIVE, Mat rvec = Mat(), Mat tvec = Mat(), Mat& reprojectionError = Mat())
    //

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns a list of all the possible solutions (a solution is a &lt;rotation vector, translation vector&gt;
     * couple), depending on the number of input points and the chosen method:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): 3 or 4 input points. Number of returned solutions can be between 0 and 4 with 3 input points.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar. Returns 2 solutions.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4 and 2 solutions are returned. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     * Only 1 solution is returned.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvecs Vector of output rotation vectors (see REF: Rodrigues ) that, together with tvecs, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvecs Vector of output translation vectors.
     * @param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param flags Method for solving a PnP problem:
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
     * F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     * @param rvec Rotation vector used to initialize an iterative PnP refinement algorithm, when flag is SOLVEPNP_ITERATIVE
     * and useExtrinsicGuess is set to true.
     * @param tvec Translation vector used to initialize an iterative PnP refinement algorithm, when flag is SOLVEPNP_ITERATIVE
     * and useExtrinsicGuess is set to true.
     * @param reprojectionError Optional vector of reprojection error, that is the RMS error
     * (\( \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} \)) between the input image points
     * and the 3D object points projected with the estimated pose.
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static int solvePnPGeneric(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, boolean useExtrinsicGuess, int flags, Mat rvec, Mat tvec, Mat reprojectionError) {
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        int retVal = solvePnPGeneric_0(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, useExtrinsicGuess, flags, rvec.nativeObj, tvec.nativeObj, reprojectionError.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns a list of all the possible solutions (a solution is a &lt;rotation vector, translation vector&gt;
     * couple), depending on the number of input points and the chosen method:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): 3 or 4 input points. Number of returned solutions can be between 0 and 4 with 3 input points.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar. Returns 2 solutions.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4 and 2 solutions are returned. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     * Only 1 solution is returned.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvecs Vector of output rotation vectors (see REF: Rodrigues ) that, together with tvecs, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvecs Vector of output translation vectors.
     * @param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param flags Method for solving a PnP problem:
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
     * F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     * @param rvec Rotation vector used to initialize an iterative PnP refinement algorithm, when flag is SOLVEPNP_ITERATIVE
     * and useExtrinsicGuess is set to true.
     * @param tvec Translation vector used to initialize an iterative PnP refinement algorithm, when flag is SOLVEPNP_ITERATIVE
     * and useExtrinsicGuess is set to true.
     * (\( \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} \)) between the input image points
     * and the 3D object points projected with the estimated pose.
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static int solvePnPGeneric(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, boolean useExtrinsicGuess, int flags, Mat rvec, Mat tvec) {
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        int retVal = solvePnPGeneric_1(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, useExtrinsicGuess, flags, rvec.nativeObj, tvec.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns a list of all the possible solutions (a solution is a &lt;rotation vector, translation vector&gt;
     * couple), depending on the number of input points and the chosen method:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): 3 or 4 input points. Number of returned solutions can be between 0 and 4 with 3 input points.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar. Returns 2 solutions.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4 and 2 solutions are returned. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     * Only 1 solution is returned.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvecs Vector of output rotation vectors (see REF: Rodrigues ) that, together with tvecs, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvecs Vector of output translation vectors.
     * @param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param flags Method for solving a PnP problem:
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
     * F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     * @param rvec Rotation vector used to initialize an iterative PnP refinement algorithm, when flag is SOLVEPNP_ITERATIVE
     * and useExtrinsicGuess is set to true.
     * and useExtrinsicGuess is set to true.
     * (\( \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} \)) between the input image points
     * and the 3D object points projected with the estimated pose.
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static int solvePnPGeneric(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, boolean useExtrinsicGuess, int flags, Mat rvec) {
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        int retVal = solvePnPGeneric_2(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, useExtrinsicGuess, flags, rvec.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns a list of all the possible solutions (a solution is a &lt;rotation vector, translation vector&gt;
     * couple), depending on the number of input points and the chosen method:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): 3 or 4 input points. Number of returned solutions can be between 0 and 4 with 3 input points.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar. Returns 2 solutions.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4 and 2 solutions are returned. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     * Only 1 solution is returned.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvecs Vector of output rotation vectors (see REF: Rodrigues ) that, together with tvecs, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvecs Vector of output translation vectors.
     * @param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * @param flags Method for solving a PnP problem:
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
     * F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     * and useExtrinsicGuess is set to true.
     * and useExtrinsicGuess is set to true.
     * (\( \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} \)) between the input image points
     * and the 3D object points projected with the estimated pose.
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static int solvePnPGeneric(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, boolean useExtrinsicGuess, int flags) {
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        int retVal = solvePnPGeneric_3(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, useExtrinsicGuess, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns a list of all the possible solutions (a solution is a &lt;rotation vector, translation vector&gt;
     * couple), depending on the number of input points and the chosen method:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): 3 or 4 input points. Number of returned solutions can be between 0 and 4 with 3 input points.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar. Returns 2 solutions.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4 and 2 solutions are returned. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     * Only 1 solution is returned.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvecs Vector of output rotation vectors (see REF: Rodrigues ) that, together with tvecs, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvecs Vector of output translation vectors.
     * @param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
     * F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     * and useExtrinsicGuess is set to true.
     * and useExtrinsicGuess is set to true.
     * (\( \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} \)) between the input image points
     * and the 3D object points projected with the estimated pose.
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static int solvePnPGeneric(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, boolean useExtrinsicGuess) {
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        int retVal = solvePnPGeneric_4(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, useExtrinsicGuess);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    /**
     * Finds an object pose from 3D-2D point correspondences.
     * This function returns a list of all the possible solutions (a solution is a &lt;rotation vector, translation vector&gt;
     * couple), depending on the number of input points and the chosen method:
     * <ul>
     *   <li>
     *  P3P methods (REF: SOLVEPNP_P3P, REF: SOLVEPNP_AP3P): 3 or 4 input points. Number of returned solutions can be between 0 and 4 with 3 input points.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE Input points must be &gt;= 4 and object points must be coplanar. Returns 2 solutions.
     *   </li>
     *   <li>
     *  REF: SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
     * Number of input points must be 4 and 2 solutions are returned. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   <li>
     *  for all the other flags, number of input points must be &gt;= 4 and object points can be in any configuration.
     * Only 1 solution is returned.
     *   </li>
     * </ul>
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
     * 1xN/Nx1 3-channel, where N is the number of points. vector&lt;Point3d&gt; can be also passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can be also passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvecs Vector of output rotation vectors (see REF: Rodrigues ) that, together with tvecs, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvecs Vector of output translation vectors.
     * the provided rvec and tvec values as initial approximations of the rotation and translation
     * vectors, respectively, and further optimizes them.
     * <ul>
     *   <li>
     *    <b>SOLVEPNP_ITERATIVE</b> Iterative method is based on a Levenberg-Marquardt optimization. In
     * this case the function finds such a pose that minimizes reprojection error, that is the sum
     * of squared distances between the observed projections imagePoints and the projected (using
     * projectPoints ) objectPoints .
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_P3P</b> Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
     * "Complete Solution Classification for the Perspective-Three-Point Problem" (CITE: gao2003complete).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_AP3P</b> Method is based on the paper of T. Ke, S. Roumeliotis
     * "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (CITE: Ke17).
     * In this case the function requires exactly four object and image points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_EPNP</b> Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
     * paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (CITE: lepetit2009epnp).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_DLS</b> Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
     * "A Direct Least-Squares (DLS) Method for PnP" (CITE: hesch2011direct).
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_UPNP</b> Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
     * F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
     * Estimation" (CITE: penate2013exhaustive). In this case the function also estimates the parameters \(f_x\) and \(f_y\)
     * assuming that both have the same value. Then the cameraMatrix is updated with the estimated
     * focal length.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE</b> Method is based on the paper of T. Collins and A. Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method requires coplanar object points.
     *   </li>
     *   <li>
     *    <b>SOLVEPNP_IPPE_SQUARE</b> Method is based on the paper of Toby Collins and Adrien Bartoli.
     * "Infinitesimal Plane-Based Pose Estimation" (CITE: Collins14). This method is suitable for marker pose estimation.
     * It requires 4 coplanar object points defined in the following order:
     *   <ul>
     *     <li>
     *    point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *    point 3: [-squareLength / 2, -squareLength / 2, 0]
     * and useExtrinsicGuess is set to true.
     * and useExtrinsicGuess is set to true.
     * (\( \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} \)) between the input image points
     * and the 3D object points projected with the estimated pose.
     *     </li>
     *   </ul>
     *
     * The function estimates the object pose given a set of object points, their corresponding image
     * projections, as well as the camera matrix and the distortion coefficients, see the figure below
     * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
     * and the Z-axis forward).
     *   </li>
     * </ul>
     *
     * ![](pnp.jpg)
     *
     * Points expressed in the world frame \( \bf{X}_w \) are projected into the image plane \( \left[ u, v \right] \)
     * using the perspective projection model \( \Pi \) and the camera intrinsic parameters matrix \( \bf{A} \):
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   u \\
     *   v \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   f_x &amp; 0 &amp; c_x \\
     *   0 &amp; f_y &amp; c_y \\
     *   0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   1 &amp; 0 &amp; 0 &amp; 0 \\
     *   0 &amp; 1 &amp; 0 &amp; 0 \\
     *   0 &amp; 0 &amp; 1 &amp; 0
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * The estimated pose is thus the rotation ({@code rvec}) and the translation ({@code tvec}) vectors that allow transforming
     * a 3D point expressed in the world frame into the camera frame:
     *
     * \(
     *   \begin{align*}
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \hspace{0.2em} ^{c}\bf{T}_w
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix} \\
     *   \begin{bmatrix}
     *   X_c \\
     *   Y_c \\
     *   Z_c \\
     *   1
     *   \end{bmatrix} &amp;=
     *   \begin{bmatrix}
     *   r_{11} &amp; r_{12} &amp; r_{13} &amp; t_x \\
     *   r_{21} &amp; r_{22} &amp; r_{23} &amp; t_y \\
     *   r_{31} &amp; r_{32} &amp; r_{33} &amp; t_z \\
     *   0 &amp; 0 &amp; 0 &amp; 1
     *   \end{bmatrix}
     *   \begin{bmatrix}
     *   X_{w} \\
     *   Y_{w} \\
     *   Z_{w} \\
     *   1
     *   \end{bmatrix}
     *   \end{align*}
     * \)
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example of how to use solvePnP for planar augmented reality can be found at
     *         opencv_source_code/samples/python/plane_ar.py
     *   </li>
     *   <li>
     *       If you are using Python:
     *   <ul>
     *     <li>
     *          Numpy array slices won't work as input because solvePnP requires contiguous
     *         arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
     *         modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *     </li>
     *     <li>
     *          The P3P algorithm requires image points to be in an array of shape (N,1,2) due
     *         to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
     *         which requires 2-channel information.
     *     </li>
     *     <li>
     *          Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
     *         it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
     *         np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
     *     </li>
     *   </ul>
     *   <li>
     *       The methods <b>SOLVEPNP_DLS</b> and <b>SOLVEPNP_UPNP</b> cannot be used as the current implementations are
     *        unstable and sometimes give completely wrong results. If you pass one of these two
     *        flags, <b>SOLVEPNP_EPNP</b> method will be used instead.
     *   </li>
     *   <li>
     *       The minimum number of points is 4 in the general case. In the case of <b>SOLVEPNP_P3P</b> and <b>SOLVEPNP_AP3P</b>
     *        methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
     *        of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_ITERATIVE</b> method and {@code useExtrinsicGuess=true}, the minimum number of points is 3 (3 points
     *        are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
     *        global solution to converge.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE</b> input points must be &gt;= 4 and object points must be coplanar.
     *   </li>
     *   <li>
     *       With <b>SOLVEPNP_IPPE_SQUARE</b> this is a special case suitable for marker pose estimation.
     *        Number of input points must be 4. Object points must be defined in the following order:
     *   <ul>
     *     <li>
     *           point 0: [-squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 1: [ squareLength / 2,  squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 2: [ squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *     <li>
     *           point 3: [-squareLength / 2, -squareLength / 2, 0]
     *     </li>
     *   </ul>
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static int solvePnPGeneric(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs) {
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        int retVal = solvePnPGeneric_5(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  void cv::Rodrigues(Mat src, Mat& dst, Mat& jacobian = Mat())
    //

    /**
     * Converts a rotation matrix to a rotation vector or vice versa.
     *
     * @param src Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).
     * @param dst Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.
     * @param jacobian Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial
     * derivatives of the output array components with respect to the input array components.
     *
     * \(\begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos(\theta) I + (1- \cos{\theta} ) r r^T +  \sin(\theta) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}\)
     *
     * Inverse transformation can be also done easily, since
     *
     * \(\sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}\)
     *
     * A rotation vector is a convenient and most compact representation of a rotation matrix (since any
     * rotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry
     * optimization procedures like REF: calibrateCamera, REF: stereoCalibrate, or REF: solvePnP .
     *
     * <b>Note:</b> More information about the computation of the derivative of a 3D rotation matrix with respect to its exponential coordinate
     * can be found in:
     * <ul>
     *   <li>
     *      A Compact Formula for the Derivative of a 3-D Rotation in Exponential Coordinates, Guillermo Gallego, Anthony J. Yezzi CITE: Gallego2014ACF
     *   </li>
     * </ul>
     *
     * <b>Note:</b> Useful information on SE(3) and Lie Groups can be found in:
     * <ul>
     *   <li>
     *      A tutorial on SE(3) transformation parameterizations and on-manifold optimization, Jose-Luis Blanco CITE: blanco2010tutorial
     *   </li>
     *   <li>
     *      Lie Groups for 2D and 3D Transformation, Ethan Eade CITE: Eade17
     *   </li>
     *   <li>
     *      A micro Lie theory for state estimation in robotics, Joan Solà, Jérémie Deray, Dinesh Atchuthan CITE: Sol2018AML
     *   </li>
     * </ul>
     */
    public static void Rodrigues(Mat src, Mat dst, Mat jacobian) {
        Rodrigues_0(src.nativeObj, dst.nativeObj, jacobian.nativeObj);
    }

    /**
     * Converts a rotation matrix to a rotation vector or vice versa.
     *
     * @param src Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).
     * @param dst Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.
     * derivatives of the output array components with respect to the input array components.
     *
     * \(\begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos(\theta) I + (1- \cos{\theta} ) r r^T +  \sin(\theta) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}\)
     *
     * Inverse transformation can be also done easily, since
     *
     * \(\sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}\)
     *
     * A rotation vector is a convenient and most compact representation of a rotation matrix (since any
     * rotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry
     * optimization procedures like REF: calibrateCamera, REF: stereoCalibrate, or REF: solvePnP .
     *
     * <b>Note:</b> More information about the computation of the derivative of a 3D rotation matrix with respect to its exponential coordinate
     * can be found in:
     * <ul>
     *   <li>
     *      A Compact Formula for the Derivative of a 3-D Rotation in Exponential Coordinates, Guillermo Gallego, Anthony J. Yezzi CITE: Gallego2014ACF
     *   </li>
     * </ul>
     *
     * <b>Note:</b> Useful information on SE(3) and Lie Groups can be found in:
     * <ul>
     *   <li>
     *      A tutorial on SE(3) transformation parameterizations and on-manifold optimization, Jose-Luis Blanco CITE: blanco2010tutorial
     *   </li>
     *   <li>
     *      Lie Groups for 2D and 3D Transformation, Ethan Eade CITE: Eade17
     *   </li>
     *   <li>
     *      A micro Lie theory for state estimation in robotics, Joan Solà, Jérémie Deray, Dinesh Atchuthan CITE: Sol2018AML
     *   </li>
     * </ul>
     */
    public static void Rodrigues(Mat src, Mat dst) {
        Rodrigues_1(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::calibrateHandEye(vector_Mat R_gripper2base, vector_Mat t_gripper2base, vector_Mat R_target2cam, vector_Mat t_target2cam, Mat& R_cam2gripper, Mat& t_cam2gripper, HandEyeCalibrationMethod method = CALIB_HAND_EYE_TSAI)
    //

    /**
     * Computes Hand-Eye calibration: \(_{}^{g}\textrm{T}_c\)
     *
     * @param R_gripper2base Rotation part extracted from the homogeneous matrix that transforms a point
     * expressed in the gripper frame to the robot base frame (\(_{}^{b}\textrm{T}_g\)).
     * This is a vector ({@code vector&lt;Mat&gt;}) that contains the rotation matrices for all the transformations
     * from gripper frame to robot base frame.
     * @param t_gripper2base Translation part extracted from the homogeneous matrix that transforms a point
     * expressed in the gripper frame to the robot base frame (\(_{}^{b}\textrm{T}_g\)).
     * This is a vector ({@code vector&lt;Mat&gt;}) that contains the translation vectors for all the transformations
     * from gripper frame to robot base frame.
     * @param R_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
     * expressed in the target frame to the camera frame (\(_{}^{c}\textrm{T}_t\)).
     * This is a vector ({@code vector&lt;Mat&gt;}) that contains the rotation matrices for all the transformations
     * from calibration target frame to camera frame.
     * @param t_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
     * expressed in the target frame to the camera frame (\(_{}^{c}\textrm{T}_t\)).
     * This is a vector ({@code vector&lt;Mat&gt;}) that contains the translation vectors for all the transformations
     * from calibration target frame to camera frame.
     * @param R_cam2gripper Estimated rotation part extracted from the homogeneous matrix that transforms a point
     * expressed in the camera frame to the gripper frame (\(_{}^{g}\textrm{T}_c\)).
     * @param t_cam2gripper Estimated translation part extracted from the homogeneous matrix that transforms a point
     * expressed in the camera frame to the gripper frame (\(_{}^{g}\textrm{T}_c\)).
     * @param method One of the implemented Hand-Eye calibration method, see cv::HandEyeCalibrationMethod
     *
     * The function performs the Hand-Eye calibration using various methods. One approach consists in estimating the
     * rotation then the translation (separable solutions) and the following methods are implemented:
     * <ul>
     *   <li>
     *    R. Tsai, R. Lenz A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/EyeCalibration \cite Tsai89
     *   </li>
     *   <li>
     *    F. Park, B. Martin Robot Sensor Calibration: Solving AX = XB on the Euclidean Group \cite Park94
     *   </li>
     *   <li>
     *    R. Horaud, F. Dornaika Hand-Eye Calibration \cite Horaud95
     *   </li>
     * </ul>
     *
     * Another approach consists in estimating simultaneously the rotation and the translation (simultaneous solutions),
     * with the following implemented method:
     * <ul>
     *   <li>
     *    N. Andreff, R. Horaud, B. Espiau On-line Hand-Eye Calibration \cite Andreff99
     *   </li>
     *   <li>
     *    K. Daniilidis Hand-Eye Calibration Using Dual Quaternions \cite Daniilidis98
     *   </li>
     * </ul>
     *
     * The following picture describes the Hand-Eye calibration problem where the transformation between a camera ("eye")
     * mounted on a robot gripper ("hand") has to be estimated.
     *
     * ![](pics/hand-eye_figure.png)
     *
     * The calibration procedure is the following:
     * <ul>
     *   <li>
     *    a static calibration pattern is used to estimate the transformation between the target frame
     *   and the camera frame
     *   </li>
     *   <li>
     *    the robot gripper is moved in order to acquire several poses
     *   </li>
     *   <li>
     *    for each pose, the homogeneous transformation between the gripper frame and the robot base frame is recorded using for
     *   instance the robot kinematics
     * \(
     *     \begin{bmatrix}
     *     X_b\\
     *     Y_b\\
     *     Z_b\\
     *     1
     *     \end{bmatrix}
     *     =
     *     \begin{bmatrix}
     *     _{}^{b}\textrm{R}_g &amp; _{}^{b}\textrm{t}_g \\
     *     0_{1 \times 3} &amp; 1
     *     \end{bmatrix}
     *     \begin{bmatrix}
     *     X_g\\
     *     Y_g\\
     *     Z_g\\
     *     1
     *     \end{bmatrix}
     * \)
     *   </li>
     *   <li>
     *    for each pose, the homogeneous transformation between the calibration target frame and the camera frame is recorded using
     *   for instance a pose estimation method (PnP) from 2D-3D point correspondences
     * \(
     *     \begin{bmatrix}
     *     X_c\\
     *     Y_c\\
     *     Z_c\\
     *     1
     *     \end{bmatrix}
     *     =
     *     \begin{bmatrix}
     *     _{}^{c}\textrm{R}_t &amp; _{}^{c}\textrm{t}_t \\
     *     0_{1 \times 3} &amp; 1
     *     \end{bmatrix}
     *     \begin{bmatrix}
     *     X_t\\
     *     Y_t\\
     *     Z_t\\
     *     1
     *     \end{bmatrix}
     * \)
     *   </li>
     * </ul>
     *
     * The Hand-Eye calibration procedure returns the following homogeneous transformation
     * \(
     *     \begin{bmatrix}
     *     X_g\\
     *     Y_g\\
     *     Z_g\\
     *     1
     *     \end{bmatrix}
     *     =
     *     \begin{bmatrix}
     *     _{}^{g}\textrm{R}_c &amp; _{}^{g}\textrm{t}_c \\
     *     0_{1 \times 3} &amp; 1
     *     \end{bmatrix}
     *     \begin{bmatrix}
     *     X_c\\
     *     Y_c\\
     *     Z_c\\
     *     1
     *     \end{bmatrix}
     * \)
     *
     * This problem is also known as solving the \(\mathbf{A}\mathbf{X}=\mathbf{X}\mathbf{B}\) equation:
     * \(
     *     \begin{align*}
     *     ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(1)} &amp;=
     *     \hspace{0.1em} ^{b}{\textrm{T}_g}^{(2)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} \\
     *
     *     (^{b}{\textrm{T}_g}^{(2)})^{-1} \hspace{0.2em} ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c &amp;=
     *     \hspace{0.1em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} (^{c}{\textrm{T}_t}^{(1)})^{-1} \\
     *
     *     \textrm{A}_i \textrm{X} &amp;= \textrm{X} \textrm{B}_i \\
     *     \end{align*}
     * \)
     *
     * \note
     * Additional information can be found on this [website](http://campar.in.tum.de/Chair/HandEyeCalibration).
     * \note
     * A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye transformation.
     * So at least 3 different poses are required, but it is strongly recommended to use many more poses.
     */
    public static void calibrateHandEye(List<Mat> R_gripper2base, List<Mat> t_gripper2base, List<Mat> R_target2cam, List<Mat> t_target2cam, Mat R_cam2gripper, Mat t_cam2gripper, int method) {
        Mat R_gripper2base_mat = Converters.vector_Mat_to_Mat(R_gripper2base);
        Mat t_gripper2base_mat = Converters.vector_Mat_to_Mat(t_gripper2base);
        Mat R_target2cam_mat = Converters.vector_Mat_to_Mat(R_target2cam);
        Mat t_target2cam_mat = Converters.vector_Mat_to_Mat(t_target2cam);
        calibrateHandEye_0(R_gripper2base_mat.nativeObj, t_gripper2base_mat.nativeObj, R_target2cam_mat.nativeObj, t_target2cam_mat.nativeObj, R_cam2gripper.nativeObj, t_cam2gripper.nativeObj, method);
    }

    /**
     * Computes Hand-Eye calibration: \(_{}^{g}\textrm{T}_c\)
     *
     * @param R_gripper2base Rotation part extracted from the homogeneous matrix that transforms a point
     * expressed in the gripper frame to the robot base frame (\(_{}^{b}\textrm{T}_g\)).
     * This is a vector ({@code vector&lt;Mat&gt;}) that contains the rotation matrices for all the transformations
     * from gripper frame to robot base frame.
     * @param t_gripper2base Translation part extracted from the homogeneous matrix that transforms a point
     * expressed in the gripper frame to the robot base frame (\(_{}^{b}\textrm{T}_g\)).
     * This is a vector ({@code vector&lt;Mat&gt;}) that contains the translation vectors for all the transformations
     * from gripper frame to robot base frame.
     * @param R_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
     * expressed in the target frame to the camera frame (\(_{}^{c}\textrm{T}_t\)).
     * This is a vector ({@code vector&lt;Mat&gt;}) that contains the rotation matrices for all the transformations
     * from calibration target frame to camera frame.
     * @param t_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
     * expressed in the target frame to the camera frame (\(_{}^{c}\textrm{T}_t\)).
     * This is a vector ({@code vector&lt;Mat&gt;}) that contains the translation vectors for all the transformations
     * from calibration target frame to camera frame.
     * @param R_cam2gripper Estimated rotation part extracted from the homogeneous matrix that transforms a point
     * expressed in the camera frame to the gripper frame (\(_{}^{g}\textrm{T}_c\)).
     * @param t_cam2gripper Estimated translation part extracted from the homogeneous matrix that transforms a point
     * expressed in the camera frame to the gripper frame (\(_{}^{g}\textrm{T}_c\)).
     *
     * The function performs the Hand-Eye calibration using various methods. One approach consists in estimating the
     * rotation then the translation (separable solutions) and the following methods are implemented:
     * <ul>
     *   <li>
     *    R. Tsai, R. Lenz A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/EyeCalibration \cite Tsai89
     *   </li>
     *   <li>
     *    F. Park, B. Martin Robot Sensor Calibration: Solving AX = XB on the Euclidean Group \cite Park94
     *   </li>
     *   <li>
     *    R. Horaud, F. Dornaika Hand-Eye Calibration \cite Horaud95
     *   </li>
     * </ul>
     *
     * Another approach consists in estimating simultaneously the rotation and the translation (simultaneous solutions),
     * with the following implemented method:
     * <ul>
     *   <li>
     *    N. Andreff, R. Horaud, B. Espiau On-line Hand-Eye Calibration \cite Andreff99
     *   </li>
     *   <li>
     *    K. Daniilidis Hand-Eye Calibration Using Dual Quaternions \cite Daniilidis98
     *   </li>
     * </ul>
     *
     * The following picture describes the Hand-Eye calibration problem where the transformation between a camera ("eye")
     * mounted on a robot gripper ("hand") has to be estimated.
     *
     * ![](pics/hand-eye_figure.png)
     *
     * The calibration procedure is the following:
     * <ul>
     *   <li>
     *    a static calibration pattern is used to estimate the transformation between the target frame
     *   and the camera frame
     *   </li>
     *   <li>
     *    the robot gripper is moved in order to acquire several poses
     *   </li>
     *   <li>
     *    for each pose, the homogeneous transformation between the gripper frame and the robot base frame is recorded using for
     *   instance the robot kinematics
     * \(
     *     \begin{bmatrix}
     *     X_b\\
     *     Y_b\\
     *     Z_b\\
     *     1
     *     \end{bmatrix}
     *     =
     *     \begin{bmatrix}
     *     _{}^{b}\textrm{R}_g &amp; _{}^{b}\textrm{t}_g \\
     *     0_{1 \times 3} &amp; 1
     *     \end{bmatrix}
     *     \begin{bmatrix}
     *     X_g\\
     *     Y_g\\
     *     Z_g\\
     *     1
     *     \end{bmatrix}
     * \)
     *   </li>
     *   <li>
     *    for each pose, the homogeneous transformation between the calibration target frame and the camera frame is recorded using
     *   for instance a pose estimation method (PnP) from 2D-3D point correspondences
     * \(
     *     \begin{bmatrix}
     *     X_c\\
     *     Y_c\\
     *     Z_c\\
     *     1
     *     \end{bmatrix}
     *     =
     *     \begin{bmatrix}
     *     _{}^{c}\textrm{R}_t &amp; _{}^{c}\textrm{t}_t \\
     *     0_{1 \times 3} &amp; 1
     *     \end{bmatrix}
     *     \begin{bmatrix}
     *     X_t\\
     *     Y_t\\
     *     Z_t\\
     *     1
     *     \end{bmatrix}
     * \)
     *   </li>
     * </ul>
     *
     * The Hand-Eye calibration procedure returns the following homogeneous transformation
     * \(
     *     \begin{bmatrix}
     *     X_g\\
     *     Y_g\\
     *     Z_g\\
     *     1
     *     \end{bmatrix}
     *     =
     *     \begin{bmatrix}
     *     _{}^{g}\textrm{R}_c &amp; _{}^{g}\textrm{t}_c \\
     *     0_{1 \times 3} &amp; 1
     *     \end{bmatrix}
     *     \begin{bmatrix}
     *     X_c\\
     *     Y_c\\
     *     Z_c\\
     *     1
     *     \end{bmatrix}
     * \)
     *
     * This problem is also known as solving the \(\mathbf{A}\mathbf{X}=\mathbf{X}\mathbf{B}\) equation:
     * \(
     *     \begin{align*}
     *     ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(1)} &amp;=
     *     \hspace{0.1em} ^{b}{\textrm{T}_g}^{(2)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} \\
     *
     *     (^{b}{\textrm{T}_g}^{(2)})^{-1} \hspace{0.2em} ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c &amp;=
     *     \hspace{0.1em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} (^{c}{\textrm{T}_t}^{(1)})^{-1} \\
     *
     *     \textrm{A}_i \textrm{X} &amp;= \textrm{X} \textrm{B}_i \\
     *     \end{align*}
     * \)
     *
     * \note
     * Additional information can be found on this [website](http://campar.in.tum.de/Chair/HandEyeCalibration).
     * \note
     * A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye transformation.
     * So at least 3 different poses are required, but it is strongly recommended to use many more poses.
     */
    public static void calibrateHandEye(List<Mat> R_gripper2base, List<Mat> t_gripper2base, List<Mat> R_target2cam, List<Mat> t_target2cam, Mat R_cam2gripper, Mat t_cam2gripper) {
        Mat R_gripper2base_mat = Converters.vector_Mat_to_Mat(R_gripper2base);
        Mat t_gripper2base_mat = Converters.vector_Mat_to_Mat(t_gripper2base);
        Mat R_target2cam_mat = Converters.vector_Mat_to_Mat(R_target2cam);
        Mat t_target2cam_mat = Converters.vector_Mat_to_Mat(t_target2cam);
        calibrateHandEye_1(R_gripper2base_mat.nativeObj, t_gripper2base_mat.nativeObj, R_target2cam_mat.nativeObj, t_target2cam_mat.nativeObj, R_cam2gripper.nativeObj, t_cam2gripper.nativeObj);
    }


    //
    // C++:  void cv::calibrationMatrixValues(Mat cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength, Point2d& principalPoint, double& aspectRatio)
    //

    /**
     * Computes useful camera characteristics from the camera matrix.
     *
     * @param cameraMatrix Input camera matrix that can be estimated by calibrateCamera or
     * stereoCalibrate .
     * @param imageSize Input image size in pixels.
     * @param apertureWidth Physical width in mm of the sensor.
     * @param apertureHeight Physical height in mm of the sensor.
     * @param fovx Output field of view in degrees along the horizontal sensor axis.
     * @param fovy Output field of view in degrees along the vertical sensor axis.
     * @param focalLength Focal length of the lens in mm.
     * @param principalPoint Principal point in mm.
     * @param aspectRatio \(f_y/f_x\)
     *
     * The function computes various useful camera characteristics from the previously estimated camera
     * matrix.
     *
     * <b>Note:</b>
     *    Do keep in mind that the unity measure 'mm' stands for whatever unit of measure one chooses for
     *     the chessboard pitch (it can thus be any value).
     */
    public static void calibrationMatrixValues(Mat cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double[] fovx, double[] fovy, double[] focalLength, Point principalPoint, double[] aspectRatio) {
        double[] fovx_out = new double[1];
        double[] fovy_out = new double[1];
        double[] focalLength_out = new double[1];
        double[] principalPoint_out = new double[2];
        double[] aspectRatio_out = new double[1];
        calibrationMatrixValues_0(cameraMatrix.nativeObj, imageSize.width, imageSize.height, apertureWidth, apertureHeight, fovx_out, fovy_out, focalLength_out, principalPoint_out, aspectRatio_out);
        if(fovx!=null) fovx[0] = (double)fovx_out[0];
        if(fovy!=null) fovy[0] = (double)fovy_out[0];
        if(focalLength!=null) focalLength[0] = (double)focalLength_out[0];
        if(principalPoint!=null){ principalPoint.x = principalPoint_out[0]; principalPoint.y = principalPoint_out[1]; } 
        if(aspectRatio!=null) aspectRatio[0] = (double)aspectRatio_out[0];
    }


    //
    // C++:  void cv::composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat& rvec3, Mat& tvec3, Mat& dr3dr1 = Mat(), Mat& dr3dt1 = Mat(), Mat& dr3dr2 = Mat(), Mat& dr3dt2 = Mat(), Mat& dt3dr1 = Mat(), Mat& dt3dt1 = Mat(), Mat& dt3dr2 = Mat(), Mat& dt3dt2 = Mat())
    //

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
     * @param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
     * @param dr3dr2 Optional output derivative of rvec3 with regard to rvec2
     * @param dr3dt2 Optional output derivative of rvec3 with regard to tvec2
     * @param dt3dr1 Optional output derivative of tvec3 with regard to rvec1
     * @param dt3dt1 Optional output derivative of tvec3 with regard to tvec1
     * @param dt3dr2 Optional output derivative of tvec3 with regard to rvec2
     * @param dt3dt2 Optional output derivative of tvec3 with regard to tvec2
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1, Mat dr3dt1, Mat dr3dr2, Mat dr3dt2, Mat dt3dr1, Mat dt3dt1, Mat dt3dr2, Mat dt3dt2) {
        composeRT_0(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj, dr3dt1.nativeObj, dr3dr2.nativeObj, dr3dt2.nativeObj, dt3dr1.nativeObj, dt3dt1.nativeObj, dt3dr2.nativeObj, dt3dt2.nativeObj);
    }

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
     * @param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
     * @param dr3dr2 Optional output derivative of rvec3 with regard to rvec2
     * @param dr3dt2 Optional output derivative of rvec3 with regard to tvec2
     * @param dt3dr1 Optional output derivative of tvec3 with regard to rvec1
     * @param dt3dt1 Optional output derivative of tvec3 with regard to tvec1
     * @param dt3dr2 Optional output derivative of tvec3 with regard to rvec2
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1, Mat dr3dt1, Mat dr3dr2, Mat dr3dt2, Mat dt3dr1, Mat dt3dt1, Mat dt3dr2) {
        composeRT_1(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj, dr3dt1.nativeObj, dr3dr2.nativeObj, dr3dt2.nativeObj, dt3dr1.nativeObj, dt3dt1.nativeObj, dt3dr2.nativeObj);
    }

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
     * @param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
     * @param dr3dr2 Optional output derivative of rvec3 with regard to rvec2
     * @param dr3dt2 Optional output derivative of rvec3 with regard to tvec2
     * @param dt3dr1 Optional output derivative of tvec3 with regard to rvec1
     * @param dt3dt1 Optional output derivative of tvec3 with regard to tvec1
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1, Mat dr3dt1, Mat dr3dr2, Mat dr3dt2, Mat dt3dr1, Mat dt3dt1) {
        composeRT_2(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj, dr3dt1.nativeObj, dr3dr2.nativeObj, dr3dt2.nativeObj, dt3dr1.nativeObj, dt3dt1.nativeObj);
    }

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
     * @param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
     * @param dr3dr2 Optional output derivative of rvec3 with regard to rvec2
     * @param dr3dt2 Optional output derivative of rvec3 with regard to tvec2
     * @param dt3dr1 Optional output derivative of tvec3 with regard to rvec1
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1, Mat dr3dt1, Mat dr3dr2, Mat dr3dt2, Mat dt3dr1) {
        composeRT_3(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj, dr3dt1.nativeObj, dr3dr2.nativeObj, dr3dt2.nativeObj, dt3dr1.nativeObj);
    }

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
     * @param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
     * @param dr3dr2 Optional output derivative of rvec3 with regard to rvec2
     * @param dr3dt2 Optional output derivative of rvec3 with regard to tvec2
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1, Mat dr3dt1, Mat dr3dr2, Mat dr3dt2) {
        composeRT_4(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj, dr3dt1.nativeObj, dr3dr2.nativeObj, dr3dt2.nativeObj);
    }

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
     * @param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
     * @param dr3dr2 Optional output derivative of rvec3 with regard to rvec2
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1, Mat dr3dt1, Mat dr3dr2) {
        composeRT_5(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj, dr3dt1.nativeObj, dr3dr2.nativeObj);
    }

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
     * @param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1, Mat dr3dt1) {
        composeRT_6(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj, dr3dt1.nativeObj);
    }

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1) {
        composeRT_7(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj);
    }

    /**
     * Combines two rotation-and-shift transformations.
     *
     * @param rvec1 First rotation vector.
     * @param tvec1 First translation vector.
     * @param rvec2 Second rotation vector.
     * @param tvec2 Second translation vector.
     * @param rvec3 Output rotation vector of the superposition.
     * @param tvec3 Output translation vector of the superposition.
     *
     * The functions compute:
     *
     * \(\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\)
     *
     * where \(\mathrm{rodrigues}\) denotes a rotation vector to a rotation matrix transformation, and
     * \(\mathrm{rodrigues}^{-1}\) denotes the inverse transformation. See Rodrigues for details.
     *
     * Also, the functions can compute the derivatives of the output vectors with regards to the input
     * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
     * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
     * function that contains a matrix multiplication.
     */
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3) {
        composeRT_8(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj);
    }


    //
    // C++:  void cv::computeCorrespondEpilines(Mat points, int whichImage, Mat F, Mat& lines)
    //

    /**
     * For points in an image of a stereo pair, computes the corresponding epilines in the other image.
     *
     * @param points Input points. \(N \times 1\) or \(1 \times N\) matrix of type CV_32FC2 or
     * vector&lt;Point2f&gt; .
     * @param whichImage Index of the image (1 or 2) that contains the points .
     * @param F Fundamental matrix that can be estimated using findFundamentalMat or stereoRectify .
     * @param lines Output vector of the epipolar lines corresponding to the points in the other image.
     * Each line \(ax + by + c=0\) is encoded by 3 numbers \((a, b, c)\) .
     *
     * For every point in one of the two images of a stereo pair, the function finds the equation of the
     * corresponding epipolar line in the other image.
     *
     * From the fundamental matrix definition (see findFundamentalMat ), line \(l^{(2)}_i\) in the second
     * image for the point \(p^{(1)}_i\) in the first image (when whichImage=1 ) is computed as:
     *
     * \(l^{(2)}_i = F p^{(1)}_i\)
     *
     * And vice versa, when whichImage=2, \(l^{(1)}_i\) is computed from \(p^{(2)}_i\) as:
     *
     * \(l^{(1)}_i = F^T p^{(2)}_i\)
     *
     * Line coefficients are defined up to a scale. They are normalized so that \(a_i^2+b_i^2=1\) .
     */
    public static void computeCorrespondEpilines(Mat points, int whichImage, Mat F, Mat lines) {
        computeCorrespondEpilines_0(points.nativeObj, whichImage, F.nativeObj, lines.nativeObj);
    }


    //
    // C++:  void cv::convertPointsFromHomogeneous(Mat src, Mat& dst)
    //

    /**
     * Converts points from homogeneous to Euclidean space.
     *
     * @param src Input vector of N-dimensional points.
     * @param dst Output vector of N-1-dimensional points.
     *
     * The function converts points homogeneous to Euclidean space using perspective projection. That is,
     * each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn). When xn=0, the
     * output point coordinates will be (0,0,0,...).
     */
    public static void convertPointsFromHomogeneous(Mat src, Mat dst) {
        convertPointsFromHomogeneous_0(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::convertPointsToHomogeneous(Mat src, Mat& dst)
    //

    /**
     * Converts points from Euclidean to homogeneous space.
     *
     * @param src Input vector of N-dimensional points.
     * @param dst Output vector of N+1-dimensional points.
     *
     * The function converts points from Euclidean to homogeneous space by appending 1's to the tuple of
     * point coordinates. That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1).
     */
    public static void convertPointsToHomogeneous(Mat src, Mat dst) {
        convertPointsToHomogeneous_0(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::correctMatches(Mat F, Mat points1, Mat points2, Mat& newPoints1, Mat& newPoints2)
    //

    /**
     * Refines coordinates of corresponding points.
     *
     * @param F 3x3 fundamental matrix.
     * @param points1 1xN array containing the first set of points.
     * @param points2 1xN array containing the second set of points.
     * @param newPoints1 The optimized points1.
     * @param newPoints2 The optimized points2.
     *
     * The function implements the Optimal Triangulation Method (see Multiple View Geometry for details).
     * For each given point correspondence points1[i] &lt;-&gt; points2[i], and a fundamental matrix F, it
     * computes the corrected correspondences newPoints1[i] &lt;-&gt; newPoints2[i] that minimize the geometric
     * error \(d(points1[i], newPoints1[i])^2 + d(points2[i],newPoints2[i])^2\) (where \(d(a,b)\) is the
     * geometric distance between points \(a\) and \(b\) ) subject to the epipolar constraint
     * \(newPoints2^T * F * newPoints1 = 0\) .
     */
    public static void correctMatches(Mat F, Mat points1, Mat points2, Mat newPoints1, Mat newPoints2) {
        correctMatches_0(F.nativeObj, points1.nativeObj, points2.nativeObj, newPoints1.nativeObj, newPoints2.nativeObj);
    }


    //
    // C++:  void cv::decomposeEssentialMat(Mat E, Mat& R1, Mat& R2, Mat& t)
    //

    /**
     * Decompose an essential matrix to possible rotations and translation.
     *
     * @param E The input essential matrix.
     * @param R1 One possible rotation matrix.
     * @param R2 Another possible rotation matrix.
     * @param t One possible translation.
     *
     * This function decomposes the essential matrix E using svd decomposition CITE: HartleyZ00. In
     * general, four possible poses exist for the decomposition of E. They are \([R_1, t]\),
     * \([R_1, -t]\), \([R_2, t]\), \([R_2, -t]\).
     *
     * If E gives the epipolar constraint \([p_2; 1]^T A^{-T} E A^{-1} [p_1; 1] = 0\) between the image
     * points \(p_1\) in the first image and \(p_2\) in second image, then any of the tuples
     * \([R_1, t]\), \([R_1, -t]\), \([R_2, t]\), \([R_2, -t]\) is a change of basis from the first
     * camera's coordinate system to the second camera's coordinate system. However, by decomposing E, one
     * can only get the direction of the translation. For this reason, the translation t is returned with
     * unit length.
     */
    public static void decomposeEssentialMat(Mat E, Mat R1, Mat R2, Mat t) {
        decomposeEssentialMat_0(E.nativeObj, R1.nativeObj, R2.nativeObj, t.nativeObj);
    }


    //
    // C++:  void cv::decomposeProjectionMatrix(Mat projMatrix, Mat& cameraMatrix, Mat& rotMatrix, Mat& transVect, Mat& rotMatrixX = Mat(), Mat& rotMatrixY = Mat(), Mat& rotMatrixZ = Mat(), Mat& eulerAngles = Mat())
    //

    /**
     * Decomposes a projection matrix into a rotation matrix and a camera matrix.
     *
     * @param projMatrix 3x4 input projection matrix P.
     * @param cameraMatrix Output 3x3 camera matrix K.
     * @param rotMatrix Output 3x3 external rotation matrix R.
     * @param transVect Output 4x1 translation vector T.
     * @param rotMatrixX Optional 3x3 rotation matrix around x-axis.
     * @param rotMatrixY Optional 3x3 rotation matrix around y-axis.
     * @param rotMatrixZ Optional 3x3 rotation matrix around z-axis.
     * @param eulerAngles Optional three-element vector containing three Euler angles of rotation in
     * degrees.
     *
     * The function computes a decomposition of a projection matrix into a calibration and a rotation
     * matrix and the position of a camera.
     *
     * It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
     * be used in OpenGL. Note, there is always more than one sequence of rotations about the three
     * principal axes that results in the same orientation of an object, e.g. see CITE: Slabaugh . Returned
     * tree rotation matrices and corresponding three Euler angles are only one of the possible solutions.
     *
     * The function is based on RQDecomp3x3 .
     */
    public static void decomposeProjectionMatrix(Mat projMatrix, Mat cameraMatrix, Mat rotMatrix, Mat transVect, Mat rotMatrixX, Mat rotMatrixY, Mat rotMatrixZ, Mat eulerAngles) {
        decomposeProjectionMatrix_0(projMatrix.nativeObj, cameraMatrix.nativeObj, rotMatrix.nativeObj, transVect.nativeObj, rotMatrixX.nativeObj, rotMatrixY.nativeObj, rotMatrixZ.nativeObj, eulerAngles.nativeObj);
    }

    /**
     * Decomposes a projection matrix into a rotation matrix and a camera matrix.
     *
     * @param projMatrix 3x4 input projection matrix P.
     * @param cameraMatrix Output 3x3 camera matrix K.
     * @param rotMatrix Output 3x3 external rotation matrix R.
     * @param transVect Output 4x1 translation vector T.
     * @param rotMatrixX Optional 3x3 rotation matrix around x-axis.
     * @param rotMatrixY Optional 3x3 rotation matrix around y-axis.
     * @param rotMatrixZ Optional 3x3 rotation matrix around z-axis.
     * degrees.
     *
     * The function computes a decomposition of a projection matrix into a calibration and a rotation
     * matrix and the position of a camera.
     *
     * It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
     * be used in OpenGL. Note, there is always more than one sequence of rotations about the three
     * principal axes that results in the same orientation of an object, e.g. see CITE: Slabaugh . Returned
     * tree rotation matrices and corresponding three Euler angles are only one of the possible solutions.
     *
     * The function is based on RQDecomp3x3 .
     */
    public static void decomposeProjectionMatrix(Mat projMatrix, Mat cameraMatrix, Mat rotMatrix, Mat transVect, Mat rotMatrixX, Mat rotMatrixY, Mat rotMatrixZ) {
        decomposeProjectionMatrix_1(projMatrix.nativeObj, cameraMatrix.nativeObj, rotMatrix.nativeObj, transVect.nativeObj, rotMatrixX.nativeObj, rotMatrixY.nativeObj, rotMatrixZ.nativeObj);
    }

    /**
     * Decomposes a projection matrix into a rotation matrix and a camera matrix.
     *
     * @param projMatrix 3x4 input projection matrix P.
     * @param cameraMatrix Output 3x3 camera matrix K.
     * @param rotMatrix Output 3x3 external rotation matrix R.
     * @param transVect Output 4x1 translation vector T.
     * @param rotMatrixX Optional 3x3 rotation matrix around x-axis.
     * @param rotMatrixY Optional 3x3 rotation matrix around y-axis.
     * degrees.
     *
     * The function computes a decomposition of a projection matrix into a calibration and a rotation
     * matrix and the position of a camera.
     *
     * It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
     * be used in OpenGL. Note, there is always more than one sequence of rotations about the three
     * principal axes that results in the same orientation of an object, e.g. see CITE: Slabaugh . Returned
     * tree rotation matrices and corresponding three Euler angles are only one of the possible solutions.
     *
     * The function is based on RQDecomp3x3 .
     */
    public static void decomposeProjectionMatrix(Mat projMatrix, Mat cameraMatrix, Mat rotMatrix, Mat transVect, Mat rotMatrixX, Mat rotMatrixY) {
        decomposeProjectionMatrix_2(projMatrix.nativeObj, cameraMatrix.nativeObj, rotMatrix.nativeObj, transVect.nativeObj, rotMatrixX.nativeObj, rotMatrixY.nativeObj);
    }

    /**
     * Decomposes a projection matrix into a rotation matrix and a camera matrix.
     *
     * @param projMatrix 3x4 input projection matrix P.
     * @param cameraMatrix Output 3x3 camera matrix K.
     * @param rotMatrix Output 3x3 external rotation matrix R.
     * @param transVect Output 4x1 translation vector T.
     * @param rotMatrixX Optional 3x3 rotation matrix around x-axis.
     * degrees.
     *
     * The function computes a decomposition of a projection matrix into a calibration and a rotation
     * matrix and the position of a camera.
     *
     * It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
     * be used in OpenGL. Note, there is always more than one sequence of rotations about the three
     * principal axes that results in the same orientation of an object, e.g. see CITE: Slabaugh . Returned
     * tree rotation matrices and corresponding three Euler angles are only one of the possible solutions.
     *
     * The function is based on RQDecomp3x3 .
     */
    public static void decomposeProjectionMatrix(Mat projMatrix, Mat cameraMatrix, Mat rotMatrix, Mat transVect, Mat rotMatrixX) {
        decomposeProjectionMatrix_3(projMatrix.nativeObj, cameraMatrix.nativeObj, rotMatrix.nativeObj, transVect.nativeObj, rotMatrixX.nativeObj);
    }

    /**
     * Decomposes a projection matrix into a rotation matrix and a camera matrix.
     *
     * @param projMatrix 3x4 input projection matrix P.
     * @param cameraMatrix Output 3x3 camera matrix K.
     * @param rotMatrix Output 3x3 external rotation matrix R.
     * @param transVect Output 4x1 translation vector T.
     * degrees.
     *
     * The function computes a decomposition of a projection matrix into a calibration and a rotation
     * matrix and the position of a camera.
     *
     * It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
     * be used in OpenGL. Note, there is always more than one sequence of rotations about the three
     * principal axes that results in the same orientation of an object, e.g. see CITE: Slabaugh . Returned
     * tree rotation matrices and corresponding three Euler angles are only one of the possible solutions.
     *
     * The function is based on RQDecomp3x3 .
     */
    public static void decomposeProjectionMatrix(Mat projMatrix, Mat cameraMatrix, Mat rotMatrix, Mat transVect) {
        decomposeProjectionMatrix_4(projMatrix.nativeObj, cameraMatrix.nativeObj, rotMatrix.nativeObj, transVect.nativeObj);
    }


    //
    // C++:  void cv::drawChessboardCorners(Mat& image, Size patternSize, vector_Point2f corners, bool patternWasFound)
    //

    /**
     * Renders the detected chessboard corners.
     *
     * @param image Destination image. It must be an 8-bit color image.
     * @param patternSize Number of inner corners per a chessboard row and column
     * (patternSize = cv::Size(points_per_row,points_per_column)).
     * @param corners Array of detected corners, the output of findChessboardCorners.
     * @param patternWasFound Parameter indicating whether the complete board was found or not. The
     * return value of findChessboardCorners should be passed here.
     *
     * The function draws individual chessboard corners detected either as red circles if the board was not
     * found, or as colored corners connected with lines if the board was found.
     */
    public static void drawChessboardCorners(Mat image, Size patternSize, MatOfPoint2f corners, boolean patternWasFound) {
        Mat corners_mat = corners;
        drawChessboardCorners_0(image.nativeObj, patternSize.width, patternSize.height, corners_mat.nativeObj, patternWasFound);
    }


    //
    // C++:  void cv::drawFrameAxes(Mat& image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, float length, int thickness = 3)
    //

    /**
     * Draw axes of the world/object coordinate system from pose estimation. SEE: solvePnP
     *
     * @param image Input/output image. It must have 1 or 3 channels. The number of channels is not altered.
     * @param cameraMatrix Input 3x3 floating-point matrix of camera intrinsic parameters.
     * \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\)
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.
     * @param rvec Rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Translation vector.
     * @param length Length of the painted axes in the same unit than tvec (usually in meters).
     * @param thickness Line thickness of the painted axes.
     *
     * This function draws the axes of the world/object coordinate system w.r.t. to the camera frame.
     * OX is drawn in red, OY in green and OZ in blue.
     */
    public static void drawFrameAxes(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, float length, int thickness) {
        drawFrameAxes_0(image.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvec.nativeObj, tvec.nativeObj, length, thickness);
    }

    /**
     * Draw axes of the world/object coordinate system from pose estimation. SEE: solvePnP
     *
     * @param image Input/output image. It must have 1 or 3 channels. The number of channels is not altered.
     * @param cameraMatrix Input 3x3 floating-point matrix of camera intrinsic parameters.
     * \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\)
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.
     * @param rvec Rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system.
     * @param tvec Translation vector.
     * @param length Length of the painted axes in the same unit than tvec (usually in meters).
     *
     * This function draws the axes of the world/object coordinate system w.r.t. to the camera frame.
     * OX is drawn in red, OY in green and OZ in blue.
     */
    public static void drawFrameAxes(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, float length) {
        drawFrameAxes_1(image.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvec.nativeObj, tvec.nativeObj, length);
    }


    //
    // C++:  void cv::filterHomographyDecompByVisibleRefpoints(vector_Mat rotations, vector_Mat normals, Mat beforePoints, Mat afterPoints, Mat& possibleSolutions, Mat pointsMask = Mat())
    //

    /**
     * Filters homography decompositions based on additional information.
     *
     * @param rotations Vector of rotation matrices.
     * @param normals Vector of plane normal matrices.
     * @param beforePoints Vector of (rectified) visible reference points before the homography is applied
     * @param afterPoints Vector of (rectified) visible reference points after the homography is applied
     * @param possibleSolutions Vector of int indices representing the viable solution set after filtering
     * @param pointsMask optional Mat/Vector of 8u type representing the mask for the inliers as given by the findHomography function
     *
     * This function is intended to filter the output of the decomposeHomographyMat based on additional
     * information as described in CITE: Malis . The summary of the method: the decomposeHomographyMat function
     * returns 2 unique solutions and their "opposites" for a total of 4 solutions. If we have access to the
     * sets of points visible in the camera frame before and after the homography transformation is applied,
     * we can determine which are the true potential solutions and which are the opposites by verifying which
     * homographies are consistent with all visible reference points being in front of the camera. The inputs
     * are left unchanged; the filtered solution set is returned as indices into the existing one.
     */
    public static void filterHomographyDecompByVisibleRefpoints(List<Mat> rotations, List<Mat> normals, Mat beforePoints, Mat afterPoints, Mat possibleSolutions, Mat pointsMask) {
        Mat rotations_mat = Converters.vector_Mat_to_Mat(rotations);
        Mat normals_mat = Converters.vector_Mat_to_Mat(normals);
        filterHomographyDecompByVisibleRefpoints_0(rotations_mat.nativeObj, normals_mat.nativeObj, beforePoints.nativeObj, afterPoints.nativeObj, possibleSolutions.nativeObj, pointsMask.nativeObj);
    }

    /**
     * Filters homography decompositions based on additional information.
     *
     * @param rotations Vector of rotation matrices.
     * @param normals Vector of plane normal matrices.
     * @param beforePoints Vector of (rectified) visible reference points before the homography is applied
     * @param afterPoints Vector of (rectified) visible reference points after the homography is applied
     * @param possibleSolutions Vector of int indices representing the viable solution set after filtering
     *
     * This function is intended to filter the output of the decomposeHomographyMat based on additional
     * information as described in CITE: Malis . The summary of the method: the decomposeHomographyMat function
     * returns 2 unique solutions and their "opposites" for a total of 4 solutions. If we have access to the
     * sets of points visible in the camera frame before and after the homography transformation is applied,
     * we can determine which are the true potential solutions and which are the opposites by verifying which
     * homographies are consistent with all visible reference points being in front of the camera. The inputs
     * are left unchanged; the filtered solution set is returned as indices into the existing one.
     */
    public static void filterHomographyDecompByVisibleRefpoints(List<Mat> rotations, List<Mat> normals, Mat beforePoints, Mat afterPoints, Mat possibleSolutions) {
        Mat rotations_mat = Converters.vector_Mat_to_Mat(rotations);
        Mat normals_mat = Converters.vector_Mat_to_Mat(normals);
        filterHomographyDecompByVisibleRefpoints_1(rotations_mat.nativeObj, normals_mat.nativeObj, beforePoints.nativeObj, afterPoints.nativeObj, possibleSolutions.nativeObj);
    }


    //
    // C++:  void cv::filterSpeckles(Mat& img, double newVal, int maxSpeckleSize, double maxDiff, Mat& buf = Mat())
    //

    /**
     * Filters off small noise blobs (speckles) in the disparity map
     *
     * @param img The input 16-bit signed disparity image
     * @param newVal The disparity value used to paint-off the speckles
     * @param maxSpeckleSize The maximum speckle size to consider it a speckle. Larger blobs are not
     * affected by the algorithm
     * @param maxDiff Maximum difference between neighbor disparity pixels to put them into the same
     * blob. Note that since StereoBM, StereoSGBM and may be other algorithms return a fixed-point
     * disparity map, where disparity values are multiplied by 16, this scale factor should be taken into
     * account when specifying this parameter value.
     * @param buf The optional temporary buffer to avoid memory allocation within the function.
     */
    public static void filterSpeckles(Mat img, double newVal, int maxSpeckleSize, double maxDiff, Mat buf) {
        filterSpeckles_0(img.nativeObj, newVal, maxSpeckleSize, maxDiff, buf.nativeObj);
    }

    /**
     * Filters off small noise blobs (speckles) in the disparity map
     *
     * @param img The input 16-bit signed disparity image
     * @param newVal The disparity value used to paint-off the speckles
     * @param maxSpeckleSize The maximum speckle size to consider it a speckle. Larger blobs are not
     * affected by the algorithm
     * @param maxDiff Maximum difference between neighbor disparity pixels to put them into the same
     * blob. Note that since StereoBM, StereoSGBM and may be other algorithms return a fixed-point
     * disparity map, where disparity values are multiplied by 16, this scale factor should be taken into
     * account when specifying this parameter value.
     */
    public static void filterSpeckles(Mat img, double newVal, int maxSpeckleSize, double maxDiff) {
        filterSpeckles_1(img.nativeObj, newVal, maxSpeckleSize, maxDiff);
    }


    //
    // C++:  void cv::initUndistortRectifyMap(Mat cameraMatrix, Mat distCoeffs, Mat R, Mat newCameraMatrix, Size size, int m1type, Mat& map1, Mat& map2)
    //

    /**
     * Computes the undistortion and rectification transformation map.
     *
     * The function computes the joint undistortion and rectification transformation and represents the
     * result in the form of maps for remap. The undistorted image looks like original, as if it is
     * captured with a camera using the camera matrix =newCameraMatrix and zero distortion. In case of a
     * monocular camera, newCameraMatrix is usually equal to cameraMatrix, or it can be computed by
     * #getOptimalNewCameraMatrix for a better control over scaling. In case of a stereo camera,
     * newCameraMatrix is normally set to P1 or P2 computed by #stereoRectify .
     *
     * Also, this new camera is oriented differently in the coordinate space, according to R. That, for
     * example, helps to align two heads of a stereo camera so that the epipolar lines on both images
     * become horizontal and have the same y- coordinate (in case of a horizontally aligned stereo camera).
     *
     * The function actually builds the maps for the inverse mapping algorithm that is used by remap. That
     * is, for each pixel \((u, v)\) in the destination (corrected and rectified) image, the function
     * computes the corresponding coordinates in the source image (that is, in the original image from
     * camera). The following process is applied:
     * \(
     * \begin{array}{l}
     * x  \leftarrow (u - {c'}_x)/{f'}_x  \\
     * y  \leftarrow (v - {c'}_y)/{f'}_y  \\
     * {[X\,Y\,W]} ^T  \leftarrow R^{-1}*[x \, y \, 1]^T  \\
     * x'  \leftarrow X/W  \\
     * y'  \leftarrow Y/W  \\
     * r^2  \leftarrow x'^2 + y'^2 \\
     * x''  \leftarrow x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
     * + 2p_1 x' y' + p_2(r^2 + 2 x'^2)  + s_1 r^2 + s_2 r^4\\
     * y''  \leftarrow y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
     * + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\
     * s\vecthree{x'''}{y'''}{1} =
     * \vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}((\tau_x, \tau_y)}
     * {0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)}
     * {0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\\
     * map_x(u,v)  \leftarrow x''' f_x + c_x  \\
     * map_y(u,v)  \leftarrow y''' f_y + c_y
     * \end{array}
     * \)
     * where \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\)
     * are the distortion coefficients.
     *
     * In case of a stereo camera, this function is called twice: once for each camera head, after
     * stereoRectify, which in its turn is called after #stereoCalibrate. But if the stereo camera
     * was not calibrated, it is still possible to compute the rectification transformations directly from
     * the fundamental matrix using #stereoRectifyUncalibrated. For each camera, the function computes
     * homography H as the rectification transformation in a pixel domain, not a rotation matrix R in 3D
     * space. R can be computed from H as
     * \(\texttt{R} = \texttt{cameraMatrix} ^{-1} \cdot \texttt{H} \cdot \texttt{cameraMatrix}\)
     * where cameraMatrix can be chosen arbitrarily.
     *
     * @param cameraMatrix Input camera matrix \(A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\)
     * of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
     * @param R Optional rectification transformation in the object space (3x3 matrix). R1 or R2 ,
     * computed by #stereoRectify can be passed here. If the matrix is empty, the identity transformation
     * is assumed. In cvInitUndistortMap R assumed to be an identity matrix.
     * @param newCameraMatrix New camera matrix \(A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}\).
     * @param size Undistorted image size.
     * @param m1type Type of the first output map that can be CV_32FC1, CV_32FC2 or CV_16SC2, see #convertMaps
     * @param map1 The first output map.
     * @param map2 The second output map.
     */
    public static void initUndistortRectifyMap(Mat cameraMatrix, Mat distCoeffs, Mat R, Mat newCameraMatrix, Size size, int m1type, Mat map1, Mat map2) {
        initUndistortRectifyMap_0(cameraMatrix.nativeObj, distCoeffs.nativeObj, R.nativeObj, newCameraMatrix.nativeObj, size.width, size.height, m1type, map1.nativeObj, map2.nativeObj);
    }


    //
    // C++:  void cv::matMulDeriv(Mat A, Mat B, Mat& dABdA, Mat& dABdB)
    //

    /**
     * Computes partial derivatives of the matrix product for each multiplied matrix.
     *
     * @param A First multiplied matrix.
     * @param B Second multiplied matrix.
     * @param dABdA First output derivative matrix d(A\*B)/dA of size
     * \(\texttt{A.rows*B.cols} \times {A.rows*A.cols}\) .
     * @param dABdB Second output derivative matrix d(A\*B)/dB of size
     * \(\texttt{A.rows*B.cols} \times {B.rows*B.cols}\) .
     *
     * The function computes partial derivatives of the elements of the matrix product \(A*B\) with regard to
     * the elements of each of the two input matrices. The function is used to compute the Jacobian
     * matrices in stereoCalibrate but can also be used in any other similar optimization function.
     */
    public static void matMulDeriv(Mat A, Mat B, Mat dABdA, Mat dABdB) {
        matMulDeriv_0(A.nativeObj, B.nativeObj, dABdA.nativeObj, dABdB.nativeObj);
    }


    //
    // C++:  void cv::projectPoints(vector_Point3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, vector_double distCoeffs, vector_Point2f& imagePoints, Mat& jacobian = Mat(), double aspectRatio = 0)
    //

    /**
     * Projects 3D points to an image plane.
     *
     * @param objectPoints Array of object points expressed wrt. the world coordinate frame. A 3xN/Nx3
     * 1-channel or 1xN/Nx1 3-channel (or vector&lt;Point3f&gt; ), where N is the number of points in the view.
     * @param rvec The rotation vector (REF: Rodrigues) that, together with tvec, performs a change of
     * basis from world to camera coordinate system, see REF: calibrateCamera for details.
     * @param tvec The translation vector, see parameter description above.
     * @param cameraMatrix Camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.
     * @param imagePoints Output array of image points, 1xN/Nx1 2-channel, or
     * vector&lt;Point2f&gt; .
     * @param jacobian Optional output 2Nx(10+&lt;numDistCoeffs&gt;) jacobian matrix of derivatives of image
     * points with respect to components of the rotation vector, translation vector, focal lengths,
     * coordinates of the principal point and the distortion coefficients. In the old interface different
     * components of the jacobian are returned via different output parameters.
     * @param aspectRatio Optional "fixed aspect ratio" parameter. If the parameter is not 0, the
     * function assumes that the aspect ratio (\(f_x / f_y\)) is fixed and correspondingly adjusts the
     * jacobian matrix.
     *
     * The function computes the 2D projections of 3D points to the image plane, given intrinsic and
     * extrinsic camera parameters. Optionally, the function computes Jacobians -matrices of partial
     * derivatives of image points coordinates (as functions of all the input parameters) with respect to
     * the particular parameters, intrinsic and/or extrinsic. The Jacobians are used during the global
     * optimization in REF: calibrateCamera, REF: solvePnP, and REF: stereoCalibrate. The function itself
     * can also be used to compute a re-projection error, given the current intrinsic and extrinsic
     * parameters.
     *
     * <b>Note:</b> By setting rvec = tvec = \([0, 0, 0]\), or by setting cameraMatrix to a 3x3 identity matrix,
     * or by passing zero distortion coefficients, one can get various useful partial cases of the
     * function. This means, one can compute the distorted coordinates for a sparse set of points or apply
     * a perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.
     */
    public static void projectPoints(MatOfPoint3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, MatOfDouble distCoeffs, MatOfPoint2f imagePoints, Mat jacobian, double aspectRatio) {
        Mat objectPoints_mat = objectPoints;
        Mat distCoeffs_mat = distCoeffs;
        Mat imagePoints_mat = imagePoints;
        projectPoints_0(objectPoints_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, imagePoints_mat.nativeObj, jacobian.nativeObj, aspectRatio);
    }

    /**
     * Projects 3D points to an image plane.
     *
     * @param objectPoints Array of object points expressed wrt. the world coordinate frame. A 3xN/Nx3
     * 1-channel or 1xN/Nx1 3-channel (or vector&lt;Point3f&gt; ), where N is the number of points in the view.
     * @param rvec The rotation vector (REF: Rodrigues) that, together with tvec, performs a change of
     * basis from world to camera coordinate system, see REF: calibrateCamera for details.
     * @param tvec The translation vector, see parameter description above.
     * @param cameraMatrix Camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.
     * @param imagePoints Output array of image points, 1xN/Nx1 2-channel, or
     * vector&lt;Point2f&gt; .
     * @param jacobian Optional output 2Nx(10+&lt;numDistCoeffs&gt;) jacobian matrix of derivatives of image
     * points with respect to components of the rotation vector, translation vector, focal lengths,
     * coordinates of the principal point and the distortion coefficients. In the old interface different
     * components of the jacobian are returned via different output parameters.
     * function assumes that the aspect ratio (\(f_x / f_y\)) is fixed and correspondingly adjusts the
     * jacobian matrix.
     *
     * The function computes the 2D projections of 3D points to the image plane, given intrinsic and
     * extrinsic camera parameters. Optionally, the function computes Jacobians -matrices of partial
     * derivatives of image points coordinates (as functions of all the input parameters) with respect to
     * the particular parameters, intrinsic and/or extrinsic. The Jacobians are used during the global
     * optimization in REF: calibrateCamera, REF: solvePnP, and REF: stereoCalibrate. The function itself
     * can also be used to compute a re-projection error, given the current intrinsic and extrinsic
     * parameters.
     *
     * <b>Note:</b> By setting rvec = tvec = \([0, 0, 0]\), or by setting cameraMatrix to a 3x3 identity matrix,
     * or by passing zero distortion coefficients, one can get various useful partial cases of the
     * function. This means, one can compute the distorted coordinates for a sparse set of points or apply
     * a perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.
     */
    public static void projectPoints(MatOfPoint3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, MatOfDouble distCoeffs, MatOfPoint2f imagePoints, Mat jacobian) {
        Mat objectPoints_mat = objectPoints;
        Mat distCoeffs_mat = distCoeffs;
        Mat imagePoints_mat = imagePoints;
        projectPoints_1(objectPoints_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, imagePoints_mat.nativeObj, jacobian.nativeObj);
    }

    /**
     * Projects 3D points to an image plane.
     *
     * @param objectPoints Array of object points expressed wrt. the world coordinate frame. A 3xN/Nx3
     * 1-channel or 1xN/Nx1 3-channel (or vector&lt;Point3f&gt; ), where N is the number of points in the view.
     * @param rvec The rotation vector (REF: Rodrigues) that, together with tvec, performs a change of
     * basis from world to camera coordinate system, see REF: calibrateCamera for details.
     * @param tvec The translation vector, see parameter description above.
     * @param cameraMatrix Camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.
     * @param imagePoints Output array of image points, 1xN/Nx1 2-channel, or
     * vector&lt;Point2f&gt; .
     * points with respect to components of the rotation vector, translation vector, focal lengths,
     * coordinates of the principal point and the distortion coefficients. In the old interface different
     * components of the jacobian are returned via different output parameters.
     * function assumes that the aspect ratio (\(f_x / f_y\)) is fixed and correspondingly adjusts the
     * jacobian matrix.
     *
     * The function computes the 2D projections of 3D points to the image plane, given intrinsic and
     * extrinsic camera parameters. Optionally, the function computes Jacobians -matrices of partial
     * derivatives of image points coordinates (as functions of all the input parameters) with respect to
     * the particular parameters, intrinsic and/or extrinsic. The Jacobians are used during the global
     * optimization in REF: calibrateCamera, REF: solvePnP, and REF: stereoCalibrate. The function itself
     * can also be used to compute a re-projection error, given the current intrinsic and extrinsic
     * parameters.
     *
     * <b>Note:</b> By setting rvec = tvec = \([0, 0, 0]\), or by setting cameraMatrix to a 3x3 identity matrix,
     * or by passing zero distortion coefficients, one can get various useful partial cases of the
     * function. This means, one can compute the distorted coordinates for a sparse set of points or apply
     * a perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.
     */
    public static void projectPoints(MatOfPoint3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, MatOfDouble distCoeffs, MatOfPoint2f imagePoints) {
        Mat objectPoints_mat = objectPoints;
        Mat distCoeffs_mat = distCoeffs;
        Mat imagePoints_mat = imagePoints;
        projectPoints_2(objectPoints_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, imagePoints_mat.nativeObj);
    }


    //
    // C++:  void cv::reprojectImageTo3D(Mat disparity, Mat& _3dImage, Mat Q, bool handleMissingValues = false, int ddepth = -1)
    //

    /**
     * Reprojects a disparity image to 3D space.
     *
     * @param disparity Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit
     * floating-point disparity image. The values of 8-bit / 16-bit signed formats are assumed to have no
     * fractional bits. If the disparity is 16-bit signed format, as computed by REF: StereoBM or
     * REF: StereoSGBM and maybe other algorithms, it should be divided by 16 (and scaled to float) before
     * being used here.
     * @param _3dImage Output 3-channel floating-point image of the same size as disparity. Each element of
     * _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map. If one
     * uses Q obtained by REF: stereoRectify, then the returned points are represented in the first
     * camera's rectified coordinate system.
     * @param Q \(4 \times 4\) perspective transformation matrix that can be obtained with
     * REF: stereoRectify.
     * @param handleMissingValues Indicates, whether the function should handle missing values (i.e.
     * points where the disparity was not computed). If handleMissingValues=true, then pixels with the
     * minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed
     * to 3D points with a very large Z value (currently set to 10000).
     * @param ddepth The optional output array depth. If it is -1, the output image will have CV_32F
     * depth. ddepth can also be set to CV_16S, CV_32S or CV_32F.
     *
     * The function transforms a single-channel disparity map to a 3-channel image representing a 3D
     * surface. That is, for each pixel (x,y) and the corresponding disparity d=disparity(x,y) , it
     * computes:
     *
     * \(\begin{bmatrix}
     * X \\
     * Y \\
     * Z \\
     * W
     * \end{bmatrix} = Q \begin{bmatrix}
     * x \\
     * y \\
     * \texttt{disparity} (x,y) \\
     * z
     * \end{bmatrix}.\)
     *
     * SEE:
     *    To reproject a sparse set of points {(x,y,d),...} to 3D space, use perspectiveTransform.
     */
    public static void reprojectImageTo3D(Mat disparity, Mat _3dImage, Mat Q, boolean handleMissingValues, int ddepth) {
        reprojectImageTo3D_0(disparity.nativeObj, _3dImage.nativeObj, Q.nativeObj, handleMissingValues, ddepth);
    }

    /**
     * Reprojects a disparity image to 3D space.
     *
     * @param disparity Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit
     * floating-point disparity image. The values of 8-bit / 16-bit signed formats are assumed to have no
     * fractional bits. If the disparity is 16-bit signed format, as computed by REF: StereoBM or
     * REF: StereoSGBM and maybe other algorithms, it should be divided by 16 (and scaled to float) before
     * being used here.
     * @param _3dImage Output 3-channel floating-point image of the same size as disparity. Each element of
     * _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map. If one
     * uses Q obtained by REF: stereoRectify, then the returned points are represented in the first
     * camera's rectified coordinate system.
     * @param Q \(4 \times 4\) perspective transformation matrix that can be obtained with
     * REF: stereoRectify.
     * @param handleMissingValues Indicates, whether the function should handle missing values (i.e.
     * points where the disparity was not computed). If handleMissingValues=true, then pixels with the
     * minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed
     * to 3D points with a very large Z value (currently set to 10000).
     * depth. ddepth can also be set to CV_16S, CV_32S or CV_32F.
     *
     * The function transforms a single-channel disparity map to a 3-channel image representing a 3D
     * surface. That is, for each pixel (x,y) and the corresponding disparity d=disparity(x,y) , it
     * computes:
     *
     * \(\begin{bmatrix}
     * X \\
     * Y \\
     * Z \\
     * W
     * \end{bmatrix} = Q \begin{bmatrix}
     * x \\
     * y \\
     * \texttt{disparity} (x,y) \\
     * z
     * \end{bmatrix}.\)
     *
     * SEE:
     *    To reproject a sparse set of points {(x,y,d),...} to 3D space, use perspectiveTransform.
     */
    public static void reprojectImageTo3D(Mat disparity, Mat _3dImage, Mat Q, boolean handleMissingValues) {
        reprojectImageTo3D_1(disparity.nativeObj, _3dImage.nativeObj, Q.nativeObj, handleMissingValues);
    }

    /**
     * Reprojects a disparity image to 3D space.
     *
     * @param disparity Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit
     * floating-point disparity image. The values of 8-bit / 16-bit signed formats are assumed to have no
     * fractional bits. If the disparity is 16-bit signed format, as computed by REF: StereoBM or
     * REF: StereoSGBM and maybe other algorithms, it should be divided by 16 (and scaled to float) before
     * being used here.
     * @param _3dImage Output 3-channel floating-point image of the same size as disparity. Each element of
     * _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map. If one
     * uses Q obtained by REF: stereoRectify, then the returned points are represented in the first
     * camera's rectified coordinate system.
     * @param Q \(4 \times 4\) perspective transformation matrix that can be obtained with
     * REF: stereoRectify.
     * points where the disparity was not computed). If handleMissingValues=true, then pixels with the
     * minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed
     * to 3D points with a very large Z value (currently set to 10000).
     * depth. ddepth can also be set to CV_16S, CV_32S or CV_32F.
     *
     * The function transforms a single-channel disparity map to a 3-channel image representing a 3D
     * surface. That is, for each pixel (x,y) and the corresponding disparity d=disparity(x,y) , it
     * computes:
     *
     * \(\begin{bmatrix}
     * X \\
     * Y \\
     * Z \\
     * W
     * \end{bmatrix} = Q \begin{bmatrix}
     * x \\
     * y \\
     * \texttt{disparity} (x,y) \\
     * z
     * \end{bmatrix}.\)
     *
     * SEE:
     *    To reproject a sparse set of points {(x,y,d),...} to 3D space, use perspectiveTransform.
     */
    public static void reprojectImageTo3D(Mat disparity, Mat _3dImage, Mat Q) {
        reprojectImageTo3D_2(disparity.nativeObj, _3dImage.nativeObj, Q.nativeObj);
    }


    //
    // C++:  void cv::solvePnPRefineLM(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat& rvec, Mat& tvec, TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, FLT_EPSILON))
    //

    /**
     * Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frame
     * to the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,
     * where N is the number of points. vector&lt;Point3d&gt; can also be passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can also be passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Input/Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system. Input values are used as an initial solution.
     * @param tvec Input/Output translation vector. Input values are used as an initial solution.
     * @param criteria Criteria when to stop the Levenberg-Marquard iterative algorithm.
     *
     * The function refines the object pose given at least 3 object points, their corresponding image
     * projections, an initial solution for the rotation and translation vector,
     * as well as the camera matrix and the distortion coefficients.
     * The function minimizes the projection error with respect to the rotation and the translation vectors, according
     * to a Levenberg-Marquardt iterative minimization CITE: Madsen04 CITE: Eade13 process.
     */
    public static void solvePnPRefineLM(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, TermCriteria criteria) {
        solvePnPRefineLM_0(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvec.nativeObj, tvec.nativeObj, criteria.type, criteria.maxCount, criteria.epsilon);
    }

    /**
     * Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frame
     * to the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,
     * where N is the number of points. vector&lt;Point3d&gt; can also be passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can also be passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Input/Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system. Input values are used as an initial solution.
     * @param tvec Input/Output translation vector. Input values are used as an initial solution.
     *
     * The function refines the object pose given at least 3 object points, their corresponding image
     * projections, an initial solution for the rotation and translation vector,
     * as well as the camera matrix and the distortion coefficients.
     * The function minimizes the projection error with respect to the rotation and the translation vectors, according
     * to a Levenberg-Marquardt iterative minimization CITE: Madsen04 CITE: Eade13 process.
     */
    public static void solvePnPRefineLM(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec) {
        solvePnPRefineLM_1(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvec.nativeObj, tvec.nativeObj);
    }


    //
    // C++:  void cv::solvePnPRefineVVS(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat& rvec, Mat& tvec, TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, FLT_EPSILON), double VVSlambda = 1)
    //

    /**
     * Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frame
     * to the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,
     * where N is the number of points. vector&lt;Point3d&gt; can also be passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can also be passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Input/Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system. Input values are used as an initial solution.
     * @param tvec Input/Output translation vector. Input values are used as an initial solution.
     * @param criteria Criteria when to stop the Levenberg-Marquard iterative algorithm.
     * @param VVSlambda Gain for the virtual visual servoing control law, equivalent to the \(\alpha\)
     * gain in the Damped Gauss-Newton formulation.
     *
     * The function refines the object pose given at least 3 object points, their corresponding image
     * projections, an initial solution for the rotation and translation vector,
     * as well as the camera matrix and the distortion coefficients.
     * The function minimizes the projection error with respect to the rotation and the translation vectors, using a
     * virtual visual servoing (VVS) CITE: Chaumette06 CITE: Marchand16 scheme.
     */
    public static void solvePnPRefineVVS(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, TermCriteria criteria, double VVSlambda) {
        solvePnPRefineVVS_0(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvec.nativeObj, tvec.nativeObj, criteria.type, criteria.maxCount, criteria.epsilon, VVSlambda);
    }

    /**
     * Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frame
     * to the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,
     * where N is the number of points. vector&lt;Point3d&gt; can also be passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can also be passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Input/Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system. Input values are used as an initial solution.
     * @param tvec Input/Output translation vector. Input values are used as an initial solution.
     * @param criteria Criteria when to stop the Levenberg-Marquard iterative algorithm.
     * gain in the Damped Gauss-Newton formulation.
     *
     * The function refines the object pose given at least 3 object points, their corresponding image
     * projections, an initial solution for the rotation and translation vector,
     * as well as the camera matrix and the distortion coefficients.
     * The function minimizes the projection error with respect to the rotation and the translation vectors, using a
     * virtual visual servoing (VVS) CITE: Chaumette06 CITE: Marchand16 scheme.
     */
    public static void solvePnPRefineVVS(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, TermCriteria criteria) {
        solvePnPRefineVVS_1(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvec.nativeObj, tvec.nativeObj, criteria.type, criteria.maxCount, criteria.epsilon);
    }

    /**
     * Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frame
     * to the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution.
     *
     * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,
     * where N is the number of points. vector&lt;Point3d&gt; can also be passed here.
     * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
     * where N is the number of points. vector&lt;Point2d&gt; can also be passed here.
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\) of
     * 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
     * assumed.
     * @param rvec Input/Output rotation vector (see REF: Rodrigues ) that, together with tvec, brings points from
     * the model coordinate system to the camera coordinate system. Input values are used as an initial solution.
     * @param tvec Input/Output translation vector. Input values are used as an initial solution.
     * gain in the Damped Gauss-Newton formulation.
     *
     * The function refines the object pose given at least 3 object points, their corresponding image
     * projections, an initial solution for the rotation and translation vector,
     * as well as the camera matrix and the distortion coefficients.
     * The function minimizes the projection error with respect to the rotation and the translation vectors, using a
     * virtual visual servoing (VVS) CITE: Chaumette06 CITE: Marchand16 scheme.
     */
    public static void solvePnPRefineVVS(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec) {
        solvePnPRefineVVS_2(objectPoints.nativeObj, imagePoints.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvec.nativeObj, tvec.nativeObj);
    }


    //
    // C++:  void cv::stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, int flags = CALIB_ZERO_DISPARITY, double alpha = -1, Size newImageSize = Size(), Rect* validPixROI1 = 0, Rect* validPixROI2 = 0)
    //

    /**
     * Computes rectification transforms for each head of a calibrated stereo camera.
     *
     * @param cameraMatrix1 First camera matrix.
     * @param distCoeffs1 First camera distortion parameters.
     * @param cameraMatrix2 Second camera matrix.
     * @param distCoeffs2 Second camera distortion parameters.
     * @param imageSize Size of the image used for stereo calibration.
     * @param R Rotation matrix from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param T Translation vector from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
     * brings points given in the unrectified first camera's coordinate system to points in the rectified
     * first camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified first camera's coordinate system to the rectified first camera's coordinate system.
     * @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
     * brings points given in the unrectified second camera's coordinate system to points in the rectified
     * second camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified second camera's coordinate system to the rectified second camera's coordinate system.
     * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified first camera's image.
     * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified second camera's image.
     * @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see REF: reprojectImageTo3D).
     * @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     * the function makes the principal points of each camera have the same pixel coordinates in the
     * rectified views. And if the flag is not set, the function may still shift the images in the
     * horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     * useful image area.
     * @param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
     * scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
     * images are zoomed and shifted so that only valid pixels are visible (no black areas after
     * rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
     * pixels from the original images from the cameras are retained in the rectified images (no source
     * image pixels are lost). Any intermediate value yields an intermediate result between
     * those two extreme cases.
     * @param newImageSize New image resolution after rectification. The same size should be passed to
     * initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     * is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
     * preserve details in the original image, especially when there is a big radial distortion.
     * @param validPixROI1 Optional output rectangles inside the rectified images where all the pixels
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     * @param validPixROI2 Optional output rectangles inside the rectified images where all the pixels
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     *
     * The function computes the rotation matrices for each camera that (virtually) make both camera image
     * planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
     * the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
     * as input. As output, it provides two rotation matrices and also two projection matrices in the new
     * coordinates. The function distinguishes the following two cases:
     *
     * <ul>
     *   <li>
     *    <b>Horizontal stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly along the x-axis (with possible small vertical shift). In the rectified images, the
     *     corresponding epipolar lines in the left and right cameras are horizontal and have the same
     *     y-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_1 &amp; 0 \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_2 &amp; T_x*f \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix} ,\)
     *
     *     where \(T_x\) is a horizontal shift between the cameras and \(cx_1=cx_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * <ul>
     *   <li>
     *    <b>Vertical stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar
     *     lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_1 &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_2 &amp; T_y*f \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix},\)
     *
     *     where \(T_y\) is a vertical shift between the cameras and \(cy_1=cy_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
     * matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
     * initialize the rectification map for each camera.
     *
     * See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
     * the corresponding image regions. This means that the images are well rectified, which is what most
     * stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
     * their interiors are all valid pixels.
     *
     * ![image](pics/stereo_undistort.jpg)
     */
    public static void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, double alpha, Size newImageSize, Rect validPixROI1, Rect validPixROI2) {
        double[] validPixROI1_out = new double[4];
        double[] validPixROI2_out = new double[4];
        stereoRectify_0(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, alpha, newImageSize.width, newImageSize.height, validPixROI1_out, validPixROI2_out);
        if(validPixROI1!=null){ validPixROI1.x = (int)validPixROI1_out[0]; validPixROI1.y = (int)validPixROI1_out[1]; validPixROI1.width = (int)validPixROI1_out[2]; validPixROI1.height = (int)validPixROI1_out[3]; } 
        if(validPixROI2!=null){ validPixROI2.x = (int)validPixROI2_out[0]; validPixROI2.y = (int)validPixROI2_out[1]; validPixROI2.width = (int)validPixROI2_out[2]; validPixROI2.height = (int)validPixROI2_out[3]; } 
    }

    /**
     * Computes rectification transforms for each head of a calibrated stereo camera.
     *
     * @param cameraMatrix1 First camera matrix.
     * @param distCoeffs1 First camera distortion parameters.
     * @param cameraMatrix2 Second camera matrix.
     * @param distCoeffs2 Second camera distortion parameters.
     * @param imageSize Size of the image used for stereo calibration.
     * @param R Rotation matrix from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param T Translation vector from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
     * brings points given in the unrectified first camera's coordinate system to points in the rectified
     * first camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified first camera's coordinate system to the rectified first camera's coordinate system.
     * @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
     * brings points given in the unrectified second camera's coordinate system to points in the rectified
     * second camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified second camera's coordinate system to the rectified second camera's coordinate system.
     * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified first camera's image.
     * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified second camera's image.
     * @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see REF: reprojectImageTo3D).
     * @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     * the function makes the principal points of each camera have the same pixel coordinates in the
     * rectified views. And if the flag is not set, the function may still shift the images in the
     * horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     * useful image area.
     * @param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
     * scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
     * images are zoomed and shifted so that only valid pixels are visible (no black areas after
     * rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
     * pixels from the original images from the cameras are retained in the rectified images (no source
     * image pixels are lost). Any intermediate value yields an intermediate result between
     * those two extreme cases.
     * @param newImageSize New image resolution after rectification. The same size should be passed to
     * initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     * is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
     * preserve details in the original image, especially when there is a big radial distortion.
     * @param validPixROI1 Optional output rectangles inside the rectified images where all the pixels
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     *
     * The function computes the rotation matrices for each camera that (virtually) make both camera image
     * planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
     * the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
     * as input. As output, it provides two rotation matrices and also two projection matrices in the new
     * coordinates. The function distinguishes the following two cases:
     *
     * <ul>
     *   <li>
     *    <b>Horizontal stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly along the x-axis (with possible small vertical shift). In the rectified images, the
     *     corresponding epipolar lines in the left and right cameras are horizontal and have the same
     *     y-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_1 &amp; 0 \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_2 &amp; T_x*f \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix} ,\)
     *
     *     where \(T_x\) is a horizontal shift between the cameras and \(cx_1=cx_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * <ul>
     *   <li>
     *    <b>Vertical stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar
     *     lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_1 &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_2 &amp; T_y*f \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix},\)
     *
     *     where \(T_y\) is a vertical shift between the cameras and \(cy_1=cy_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
     * matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
     * initialize the rectification map for each camera.
     *
     * See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
     * the corresponding image regions. This means that the images are well rectified, which is what most
     * stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
     * their interiors are all valid pixels.
     *
     * ![image](pics/stereo_undistort.jpg)
     */
    public static void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, double alpha, Size newImageSize, Rect validPixROI1) {
        double[] validPixROI1_out = new double[4];
        stereoRectify_1(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, alpha, newImageSize.width, newImageSize.height, validPixROI1_out);
        if(validPixROI1!=null){ validPixROI1.x = (int)validPixROI1_out[0]; validPixROI1.y = (int)validPixROI1_out[1]; validPixROI1.width = (int)validPixROI1_out[2]; validPixROI1.height = (int)validPixROI1_out[3]; } 
    }

    /**
     * Computes rectification transforms for each head of a calibrated stereo camera.
     *
     * @param cameraMatrix1 First camera matrix.
     * @param distCoeffs1 First camera distortion parameters.
     * @param cameraMatrix2 Second camera matrix.
     * @param distCoeffs2 Second camera distortion parameters.
     * @param imageSize Size of the image used for stereo calibration.
     * @param R Rotation matrix from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param T Translation vector from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
     * brings points given in the unrectified first camera's coordinate system to points in the rectified
     * first camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified first camera's coordinate system to the rectified first camera's coordinate system.
     * @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
     * brings points given in the unrectified second camera's coordinate system to points in the rectified
     * second camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified second camera's coordinate system to the rectified second camera's coordinate system.
     * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified first camera's image.
     * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified second camera's image.
     * @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see REF: reprojectImageTo3D).
     * @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     * the function makes the principal points of each camera have the same pixel coordinates in the
     * rectified views. And if the flag is not set, the function may still shift the images in the
     * horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     * useful image area.
     * @param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
     * scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
     * images are zoomed and shifted so that only valid pixels are visible (no black areas after
     * rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
     * pixels from the original images from the cameras are retained in the rectified images (no source
     * image pixels are lost). Any intermediate value yields an intermediate result between
     * those two extreme cases.
     * @param newImageSize New image resolution after rectification. The same size should be passed to
     * initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     * is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
     * preserve details in the original image, especially when there is a big radial distortion.
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     *
     * The function computes the rotation matrices for each camera that (virtually) make both camera image
     * planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
     * the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
     * as input. As output, it provides two rotation matrices and also two projection matrices in the new
     * coordinates. The function distinguishes the following two cases:
     *
     * <ul>
     *   <li>
     *    <b>Horizontal stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly along the x-axis (with possible small vertical shift). In the rectified images, the
     *     corresponding epipolar lines in the left and right cameras are horizontal and have the same
     *     y-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_1 &amp; 0 \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_2 &amp; T_x*f \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix} ,\)
     *
     *     where \(T_x\) is a horizontal shift between the cameras and \(cx_1=cx_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * <ul>
     *   <li>
     *    <b>Vertical stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar
     *     lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_1 &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_2 &amp; T_y*f \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix},\)
     *
     *     where \(T_y\) is a vertical shift between the cameras and \(cy_1=cy_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
     * matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
     * initialize the rectification map for each camera.
     *
     * See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
     * the corresponding image regions. This means that the images are well rectified, which is what most
     * stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
     * their interiors are all valid pixels.
     *
     * ![image](pics/stereo_undistort.jpg)
     */
    public static void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, double alpha, Size newImageSize) {
        stereoRectify_2(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, alpha, newImageSize.width, newImageSize.height);
    }

    /**
     * Computes rectification transforms for each head of a calibrated stereo camera.
     *
     * @param cameraMatrix1 First camera matrix.
     * @param distCoeffs1 First camera distortion parameters.
     * @param cameraMatrix2 Second camera matrix.
     * @param distCoeffs2 Second camera distortion parameters.
     * @param imageSize Size of the image used for stereo calibration.
     * @param R Rotation matrix from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param T Translation vector from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
     * brings points given in the unrectified first camera's coordinate system to points in the rectified
     * first camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified first camera's coordinate system to the rectified first camera's coordinate system.
     * @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
     * brings points given in the unrectified second camera's coordinate system to points in the rectified
     * second camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified second camera's coordinate system to the rectified second camera's coordinate system.
     * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified first camera's image.
     * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified second camera's image.
     * @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see REF: reprojectImageTo3D).
     * @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     * the function makes the principal points of each camera have the same pixel coordinates in the
     * rectified views. And if the flag is not set, the function may still shift the images in the
     * horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     * useful image area.
     * @param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
     * scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
     * images are zoomed and shifted so that only valid pixels are visible (no black areas after
     * rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
     * pixels from the original images from the cameras are retained in the rectified images (no source
     * image pixels are lost). Any intermediate value yields an intermediate result between
     * those two extreme cases.
     * initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     * is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
     * preserve details in the original image, especially when there is a big radial distortion.
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     *
     * The function computes the rotation matrices for each camera that (virtually) make both camera image
     * planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
     * the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
     * as input. As output, it provides two rotation matrices and also two projection matrices in the new
     * coordinates. The function distinguishes the following two cases:
     *
     * <ul>
     *   <li>
     *    <b>Horizontal stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly along the x-axis (with possible small vertical shift). In the rectified images, the
     *     corresponding epipolar lines in the left and right cameras are horizontal and have the same
     *     y-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_1 &amp; 0 \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_2 &amp; T_x*f \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix} ,\)
     *
     *     where \(T_x\) is a horizontal shift between the cameras and \(cx_1=cx_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * <ul>
     *   <li>
     *    <b>Vertical stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar
     *     lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_1 &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_2 &amp; T_y*f \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix},\)
     *
     *     where \(T_y\) is a vertical shift between the cameras and \(cy_1=cy_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
     * matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
     * initialize the rectification map for each camera.
     *
     * See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
     * the corresponding image regions. This means that the images are well rectified, which is what most
     * stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
     * their interiors are all valid pixels.
     *
     * ![image](pics/stereo_undistort.jpg)
     */
    public static void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, double alpha) {
        stereoRectify_3(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, alpha);
    }

    /**
     * Computes rectification transforms for each head of a calibrated stereo camera.
     *
     * @param cameraMatrix1 First camera matrix.
     * @param distCoeffs1 First camera distortion parameters.
     * @param cameraMatrix2 Second camera matrix.
     * @param distCoeffs2 Second camera distortion parameters.
     * @param imageSize Size of the image used for stereo calibration.
     * @param R Rotation matrix from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param T Translation vector from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
     * brings points given in the unrectified first camera's coordinate system to points in the rectified
     * first camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified first camera's coordinate system to the rectified first camera's coordinate system.
     * @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
     * brings points given in the unrectified second camera's coordinate system to points in the rectified
     * second camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified second camera's coordinate system to the rectified second camera's coordinate system.
     * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified first camera's image.
     * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified second camera's image.
     * @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see REF: reprojectImageTo3D).
     * @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     * the function makes the principal points of each camera have the same pixel coordinates in the
     * rectified views. And if the flag is not set, the function may still shift the images in the
     * horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     * useful image area.
     * scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
     * images are zoomed and shifted so that only valid pixels are visible (no black areas after
     * rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
     * pixels from the original images from the cameras are retained in the rectified images (no source
     * image pixels are lost). Any intermediate value yields an intermediate result between
     * those two extreme cases.
     * initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     * is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
     * preserve details in the original image, especially when there is a big radial distortion.
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     *
     * The function computes the rotation matrices for each camera that (virtually) make both camera image
     * planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
     * the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
     * as input. As output, it provides two rotation matrices and also two projection matrices in the new
     * coordinates. The function distinguishes the following two cases:
     *
     * <ul>
     *   <li>
     *    <b>Horizontal stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly along the x-axis (with possible small vertical shift). In the rectified images, the
     *     corresponding epipolar lines in the left and right cameras are horizontal and have the same
     *     y-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_1 &amp; 0 \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_2 &amp; T_x*f \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix} ,\)
     *
     *     where \(T_x\) is a horizontal shift between the cameras and \(cx_1=cx_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * <ul>
     *   <li>
     *    <b>Vertical stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar
     *     lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_1 &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_2 &amp; T_y*f \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix},\)
     *
     *     where \(T_y\) is a vertical shift between the cameras and \(cy_1=cy_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
     * matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
     * initialize the rectification map for each camera.
     *
     * See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
     * the corresponding image regions. This means that the images are well rectified, which is what most
     * stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
     * their interiors are all valid pixels.
     *
     * ![image](pics/stereo_undistort.jpg)
     */
    public static void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags) {
        stereoRectify_4(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags);
    }

    /**
     * Computes rectification transforms for each head of a calibrated stereo camera.
     *
     * @param cameraMatrix1 First camera matrix.
     * @param distCoeffs1 First camera distortion parameters.
     * @param cameraMatrix2 Second camera matrix.
     * @param distCoeffs2 Second camera distortion parameters.
     * @param imageSize Size of the image used for stereo calibration.
     * @param R Rotation matrix from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param T Translation vector from the coordinate system of the first camera to the second camera,
     * see REF: stereoCalibrate.
     * @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
     * brings points given in the unrectified first camera's coordinate system to points in the rectified
     * first camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified first camera's coordinate system to the rectified first camera's coordinate system.
     * @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
     * brings points given in the unrectified second camera's coordinate system to points in the rectified
     * second camera's coordinate system. In more technical terms, it performs a change of basis from the
     * unrectified second camera's coordinate system to the rectified second camera's coordinate system.
     * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified first camera's image.
     * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     * camera, i.e. it projects points given in the rectified first camera coordinate system into the
     * rectified second camera's image.
     * @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see REF: reprojectImageTo3D).
     * the function makes the principal points of each camera have the same pixel coordinates in the
     * rectified views. And if the flag is not set, the function may still shift the images in the
     * horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     * useful image area.
     * scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
     * images are zoomed and shifted so that only valid pixels are visible (no black areas after
     * rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
     * pixels from the original images from the cameras are retained in the rectified images (no source
     * image pixels are lost). Any intermediate value yields an intermediate result between
     * those two extreme cases.
     * initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     * is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
     * preserve details in the original image, especially when there is a big radial distortion.
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     * are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
     * (see the picture below).
     *
     * The function computes the rotation matrices for each camera that (virtually) make both camera image
     * planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
     * the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
     * as input. As output, it provides two rotation matrices and also two projection matrices in the new
     * coordinates. The function distinguishes the following two cases:
     *
     * <ul>
     *   <li>
     *    <b>Horizontal stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly along the x-axis (with possible small vertical shift). In the rectified images, the
     *     corresponding epipolar lines in the left and right cameras are horizontal and have the same
     *     y-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_1 &amp; 0 \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx_2 &amp; T_x*f \\
     *                         0 &amp; f &amp; cy &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix} ,\)
     *
     *     where \(T_x\) is a horizontal shift between the cameras and \(cx_1=cx_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * <ul>
     *   <li>
     *    <b>Vertical stereo</b>: the first and the second camera views are shifted relative to each other
     *     mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar
     *     lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:
     *   </li>
     * </ul>
     *
     *     \(\texttt{P1} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_1 &amp; 0 \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix}\)
     *
     *     \(\texttt{P2} = \begin{bmatrix}
     *                         f &amp; 0 &amp; cx &amp; 0 \\
     *                         0 &amp; f &amp; cy_2 &amp; T_y*f \\
     *                         0 &amp; 0 &amp; 1 &amp; 0
     *                      \end{bmatrix},\)
     *
     *     where \(T_y\) is a vertical shift between the cameras and \(cy_1=cy_2\) if
     *     CALIB_ZERO_DISPARITY is set.
     *
     * As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
     * matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
     * initialize the rectification map for each camera.
     *
     * See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
     * the corresponding image regions. This means that the images are well rectified, which is what most
     * stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
     * their interiors are all valid pixels.
     *
     * ![image](pics/stereo_undistort.jpg)
     */
    public static void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q) {
        stereoRectify_5(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj);
    }


    //
    // C++:  void cv::triangulatePoints(Mat projMatr1, Mat projMatr2, Mat projPoints1, Mat projPoints2, Mat& points4D)
    //

    /**
     * This function reconstructs 3-dimensional points (in homogeneous coordinates) by using
     * their observations with a stereo camera.
     *
     * @param projMatr1 3x4 projection matrix of the first camera, i.e. this matrix projects 3D points
     * given in the world's coordinate system into the first image.
     * @param projMatr2 3x4 projection matrix of the second camera, i.e. this matrix projects 3D points
     * given in the world's coordinate system into the second image.
     * @param projPoints1 2xN array of feature points in the first image. In the case of the c++ version,
     * it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
     * @param projPoints2 2xN array of corresponding points in the second image. In the case of the c++
     * version, it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
     * @param points4D 4xN array of reconstructed points in homogeneous coordinates. These points are
     * returned in the world's coordinate system.
     *
     * <b>Note:</b>
     *    Keep in mind that all input data should be of float type in order for this function to work.
     *
     * <b>Note:</b>
     *    If the projection matrices from REF: stereoRectify are used, then the returned points are
     *    represented in the first camera's rectified coordinate system.
     *
     * SEE:
     *    reprojectImageTo3D
     */
    public static void triangulatePoints(Mat projMatr1, Mat projMatr2, Mat projPoints1, Mat projPoints2, Mat points4D) {
        triangulatePoints_0(projMatr1.nativeObj, projMatr2.nativeObj, projPoints1.nativeObj, projPoints2.nativeObj, points4D.nativeObj);
    }


    //
    // C++:  void cv::undistort(Mat src, Mat& dst, Mat cameraMatrix, Mat distCoeffs, Mat newCameraMatrix = Mat())
    //

    /**
     * Transforms an image to compensate for lens distortion.
     *
     * The function transforms an image to compensate radial and tangential lens distortion.
     *
     * The function is simply a combination of #initUndistortRectifyMap (with unity R ) and #remap
     * (with bilinear interpolation). See the former function for details of the transformation being
     * performed.
     *
     * Those pixels in the destination image, for which there is no correspondent pixels in the source
     * image, are filled with zeros (black color).
     *
     * A particular subset of the source image that will be visible in the corrected image can be regulated
     * by newCameraMatrix. You can use #getOptimalNewCameraMatrix to compute the appropriate
     * newCameraMatrix depending on your requirements.
     *
     * The camera matrix and the distortion parameters can be determined using #calibrateCamera. If
     * the resolution of images is different from the resolution used at the calibration stage, \(f_x,
     * f_y, c_x\) and \(c_y\) need to be scaled accordingly, while the distortion coefficients remain
     * the same.
     *
     * @param src Input (distorted) image.
     * @param dst Output (corrected) image that has the same size and type as src .
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\)
     * of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
     * @param newCameraMatrix Camera matrix of the distorted image. By default, it is the same as
     * cameraMatrix but you may additionally scale and shift the result by using a different matrix.
     */
    public static void undistort(Mat src, Mat dst, Mat cameraMatrix, Mat distCoeffs, Mat newCameraMatrix) {
        undistort_0(src.nativeObj, dst.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, newCameraMatrix.nativeObj);
    }

    /**
     * Transforms an image to compensate for lens distortion.
     *
     * The function transforms an image to compensate radial and tangential lens distortion.
     *
     * The function is simply a combination of #initUndistortRectifyMap (with unity R ) and #remap
     * (with bilinear interpolation). See the former function for details of the transformation being
     * performed.
     *
     * Those pixels in the destination image, for which there is no correspondent pixels in the source
     * image, are filled with zeros (black color).
     *
     * A particular subset of the source image that will be visible in the corrected image can be regulated
     * by newCameraMatrix. You can use #getOptimalNewCameraMatrix to compute the appropriate
     * newCameraMatrix depending on your requirements.
     *
     * The camera matrix and the distortion parameters can be determined using #calibrateCamera. If
     * the resolution of images is different from the resolution used at the calibration stage, \(f_x,
     * f_y, c_x\) and \(c_y\) need to be scaled accordingly, while the distortion coefficients remain
     * the same.
     *
     * @param src Input (distorted) image.
     * @param dst Output (corrected) image that has the same size and type as src .
     * @param cameraMatrix Input camera matrix \(A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\)
     * of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
     * cameraMatrix but you may additionally scale and shift the result by using a different matrix.
     */
    public static void undistort(Mat src, Mat dst, Mat cameraMatrix, Mat distCoeffs) {
        undistort_1(src.nativeObj, dst.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj);
    }


    //
    // C++:  void cv::undistortPoints(Mat src, Mat& dst, Mat cameraMatrix, Mat distCoeffs, Mat R, Mat P, TermCriteria criteria)
    //

    /**
     *
     *     <b>Note:</b> Default version of #undistortPoints does 5 iterations to compute undistorted points.
     * @param src automatically generated
     * @param dst automatically generated
     * @param cameraMatrix automatically generated
     * @param distCoeffs automatically generated
     * @param R automatically generated
     * @param P automatically generated
     * @param criteria automatically generated
     */
    public static void undistortPointsIter(Mat src, Mat dst, Mat cameraMatrix, Mat distCoeffs, Mat R, Mat P, TermCriteria criteria) {
        undistortPointsIter_0(src.nativeObj, dst.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, R.nativeObj, P.nativeObj, criteria.type, criteria.maxCount, criteria.epsilon);
    }


    //
    // C++:  void cv::undistortPoints(vector_Point2f src, vector_Point2f& dst, Mat cameraMatrix, Mat distCoeffs, Mat R = Mat(), Mat P = Mat())
    //

    /**
     * Computes the ideal point coordinates from the observed point coordinates.
     *
     * The function is similar to #undistort and #initUndistortRectifyMap but it operates on a
     * sparse set of points instead of a raster image. Also the function performs a reverse transformation
     * to projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a
     * planar object, it does, up to a translation vector, if the proper R is specified.
     *
     * For each observed point coordinate \((u, v)\) the function computes:
     * \(
     * \begin{array}{l}
     * x^{"}  \leftarrow (u - c_x)/f_x  \\
     * y^{"}  \leftarrow (v - c_y)/f_y  \\
     * (x',y') = undistort(x^{"},y^{"}, \texttt{distCoeffs}) \\
     * {[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\
     * x  \leftarrow X/W  \\
     * y  \leftarrow Y/W  \\
     * \text{only performed if P is specified:} \\
     * u'  \leftarrow x {f'}_x + {c'}_x  \\
     * v'  \leftarrow y {f'}_y + {c'}_y
     * \end{array}
     * \)
     *
     * where *undistort* is an approximate iterative algorithm that estimates the normalized original
     * point coordinates out of the normalized distorted point coordinates ("normalized" means that the
     * coordinates do not depend on the camera matrix).
     *
     * The function can be used for both a stereo camera head or a monocular camera (when R is empty).
     * @param src Observed point coordinates, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2) (or
     * vector&lt;Point2f&gt; ).
     * @param dst Output ideal point coordinates (1xN/Nx1 2-channel or vector&lt;Point2f&gt; ) after undistortion and reverse perspective
     * transformation. If matrix P is identity or omitted, dst will contain normalized point coordinates.
     * @param cameraMatrix Camera matrix \(\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\)
     * of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
     * @param R Rectification transformation in the object space (3x3 matrix). R1 or R2 computed by
     * #stereoRectify can be passed here. If the matrix is empty, the identity transformation is used.
     * @param P New camera matrix (3x3) or new projection matrix (3x4) \(\begin{bmatrix} {f'}_x &amp; 0 &amp; {c'}_x &amp; t_x \\ 0 &amp; {f'}_y &amp; {c'}_y &amp; t_y \\ 0 &amp; 0 &amp; 1 &amp; t_z \end{bmatrix}\). P1 or P2 computed by
     * #stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used.
     */
    public static void undistortPoints(MatOfPoint2f src, MatOfPoint2f dst, Mat cameraMatrix, Mat distCoeffs, Mat R, Mat P) {
        Mat src_mat = src;
        Mat dst_mat = dst;
        undistortPoints_0(src_mat.nativeObj, dst_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, R.nativeObj, P.nativeObj);
    }

    /**
     * Computes the ideal point coordinates from the observed point coordinates.
     *
     * The function is similar to #undistort and #initUndistortRectifyMap but it operates on a
     * sparse set of points instead of a raster image. Also the function performs a reverse transformation
     * to projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a
     * planar object, it does, up to a translation vector, if the proper R is specified.
     *
     * For each observed point coordinate \((u, v)\) the function computes:
     * \(
     * \begin{array}{l}
     * x^{"}  \leftarrow (u - c_x)/f_x  \\
     * y^{"}  \leftarrow (v - c_y)/f_y  \\
     * (x',y') = undistort(x^{"},y^{"}, \texttt{distCoeffs}) \\
     * {[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\
     * x  \leftarrow X/W  \\
     * y  \leftarrow Y/W  \\
     * \text{only performed if P is specified:} \\
     * u'  \leftarrow x {f'}_x + {c'}_x  \\
     * v'  \leftarrow y {f'}_y + {c'}_y
     * \end{array}
     * \)
     *
     * where *undistort* is an approximate iterative algorithm that estimates the normalized original
     * point coordinates out of the normalized distorted point coordinates ("normalized" means that the
     * coordinates do not depend on the camera matrix).
     *
     * The function can be used for both a stereo camera head or a monocular camera (when R is empty).
     * @param src Observed point coordinates, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2) (or
     * vector&lt;Point2f&gt; ).
     * @param dst Output ideal point coordinates (1xN/Nx1 2-channel or vector&lt;Point2f&gt; ) after undistortion and reverse perspective
     * transformation. If matrix P is identity or omitted, dst will contain normalized point coordinates.
     * @param cameraMatrix Camera matrix \(\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\)
     * of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
     * @param R Rectification transformation in the object space (3x3 matrix). R1 or R2 computed by
     * #stereoRectify can be passed here. If the matrix is empty, the identity transformation is used.
     * #stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used.
     */
    public static void undistortPoints(MatOfPoint2f src, MatOfPoint2f dst, Mat cameraMatrix, Mat distCoeffs, Mat R) {
        Mat src_mat = src;
        Mat dst_mat = dst;
        undistortPoints_1(src_mat.nativeObj, dst_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj, R.nativeObj);
    }

    /**
     * Computes the ideal point coordinates from the observed point coordinates.
     *
     * The function is similar to #undistort and #initUndistortRectifyMap but it operates on a
     * sparse set of points instead of a raster image. Also the function performs a reverse transformation
     * to projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a
     * planar object, it does, up to a translation vector, if the proper R is specified.
     *
     * For each observed point coordinate \((u, v)\) the function computes:
     * \(
     * \begin{array}{l}
     * x^{"}  \leftarrow (u - c_x)/f_x  \\
     * y^{"}  \leftarrow (v - c_y)/f_y  \\
     * (x',y') = undistort(x^{"},y^{"}, \texttt{distCoeffs}) \\
     * {[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\
     * x  \leftarrow X/W  \\
     * y  \leftarrow Y/W  \\
     * \text{only performed if P is specified:} \\
     * u'  \leftarrow x {f'}_x + {c'}_x  \\
     * v'  \leftarrow y {f'}_y + {c'}_y
     * \end{array}
     * \)
     *
     * where *undistort* is an approximate iterative algorithm that estimates the normalized original
     * point coordinates out of the normalized distorted point coordinates ("normalized" means that the
     * coordinates do not depend on the camera matrix).
     *
     * The function can be used for both a stereo camera head or a monocular camera (when R is empty).
     * @param src Observed point coordinates, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2) (or
     * vector&lt;Point2f&gt; ).
     * @param dst Output ideal point coordinates (1xN/Nx1 2-channel or vector&lt;Point2f&gt; ) after undistortion and reverse perspective
     * transformation. If matrix P is identity or omitted, dst will contain normalized point coordinates.
     * @param cameraMatrix Camera matrix \(\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\) .
     * @param distCoeffs Input vector of distortion coefficients
     * \((k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\)
     * of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
     * #stereoRectify can be passed here. If the matrix is empty, the identity transformation is used.
     * #stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used.
     */
    public static void undistortPoints(MatOfPoint2f src, MatOfPoint2f dst, Mat cameraMatrix, Mat distCoeffs) {
        Mat src_mat = src;
        Mat dst_mat = dst;
        undistortPoints_2(src_mat.nativeObj, dst_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs.nativeObj);
    }


    //
    // C++:  void cv::validateDisparity(Mat& disparity, Mat cost, int minDisparity, int numberOfDisparities, int disp12MaxDisp = 1)
    //

    public static void validateDisparity(Mat disparity, Mat cost, int minDisparity, int numberOfDisparities, int disp12MaxDisp) {
        validateDisparity_0(disparity.nativeObj, cost.nativeObj, minDisparity, numberOfDisparities, disp12MaxDisp);
    }

    public static void validateDisparity(Mat disparity, Mat cost, int minDisparity, int numberOfDisparities) {
        validateDisparity_1(disparity.nativeObj, cost.nativeObj, minDisparity, numberOfDisparities);
    }


    //
    // C++:  void cv::fisheye::distortPoints(Mat undistorted, Mat& distorted, Mat K, Mat D, double alpha = 0)
    //

    /**
     * Distorts 2D points using fisheye model.
     *
     *     @param undistorted Array of object points, 1xN/Nx1 2-channel (or vector&lt;Point2f&gt; ), where N is
     *     the number of points in the view.
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param alpha The skew coefficient.
     *     @param distorted Output array of image points, 1xN/Nx1 2-channel, or vector&lt;Point2f&gt; .
     *
     *     Note that the function assumes the camera matrix of the undistorted points to be identity.
     *     This means if you want to transform back points undistorted with undistortPoints() you have to
     *     multiply them with \(P^{-1}\).
     */
    public static void fisheye_distortPoints(Mat undistorted, Mat distorted, Mat K, Mat D, double alpha) {
        fisheye_distortPoints_0(undistorted.nativeObj, distorted.nativeObj, K.nativeObj, D.nativeObj, alpha);
    }

    /**
     * Distorts 2D points using fisheye model.
     *
     *     @param undistorted Array of object points, 1xN/Nx1 2-channel (or vector&lt;Point2f&gt; ), where N is
     *     the number of points in the view.
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param distorted Output array of image points, 1xN/Nx1 2-channel, or vector&lt;Point2f&gt; .
     *
     *     Note that the function assumes the camera matrix of the undistorted points to be identity.
     *     This means if you want to transform back points undistorted with undistortPoints() you have to
     *     multiply them with \(P^{-1}\).
     */
    public static void fisheye_distortPoints(Mat undistorted, Mat distorted, Mat K, Mat D) {
        fisheye_distortPoints_1(undistorted.nativeObj, distorted.nativeObj, K.nativeObj, D.nativeObj);
    }


    //
    // C++:  void cv::fisheye::estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat& P, double balance = 0.0, Size new_size = Size(), double fov_scale = 1.0)
    //

    /**
     * Estimates new camera matrix for undistortion or rectification.
     *
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param image_size Size of the image
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
     *     1-channel or 1x1 3-channel
     *     @param P New camera matrix (3x3) or new projection matrix (3x4)
     *     @param balance Sets the new focal length in range between the min focal length and the max focal
     *     length. Balance is in range of [0, 1].
     *     @param new_size the new size
     *     @param fov_scale Divisor for new focal length.
     */
    public static void fisheye_estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat P, double balance, Size new_size, double fov_scale) {
        fisheye_estimateNewCameraMatrixForUndistortRectify_0(K.nativeObj, D.nativeObj, image_size.width, image_size.height, R.nativeObj, P.nativeObj, balance, new_size.width, new_size.height, fov_scale);
    }

    /**
     * Estimates new camera matrix for undistortion or rectification.
     *
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param image_size Size of the image
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
     *     1-channel or 1x1 3-channel
     *     @param P New camera matrix (3x3) or new projection matrix (3x4)
     *     @param balance Sets the new focal length in range between the min focal length and the max focal
     *     length. Balance is in range of [0, 1].
     *     @param new_size the new size
     */
    public static void fisheye_estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat P, double balance, Size new_size) {
        fisheye_estimateNewCameraMatrixForUndistortRectify_1(K.nativeObj, D.nativeObj, image_size.width, image_size.height, R.nativeObj, P.nativeObj, balance, new_size.width, new_size.height);
    }

    /**
     * Estimates new camera matrix for undistortion or rectification.
     *
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param image_size Size of the image
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
     *     1-channel or 1x1 3-channel
     *     @param P New camera matrix (3x3) or new projection matrix (3x4)
     *     @param balance Sets the new focal length in range between the min focal length and the max focal
     *     length. Balance is in range of [0, 1].
     */
    public static void fisheye_estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat P, double balance) {
        fisheye_estimateNewCameraMatrixForUndistortRectify_2(K.nativeObj, D.nativeObj, image_size.width, image_size.height, R.nativeObj, P.nativeObj, balance);
    }

    /**
     * Estimates new camera matrix for undistortion or rectification.
     *
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param image_size Size of the image
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
     *     1-channel or 1x1 3-channel
     *     @param P New camera matrix (3x3) or new projection matrix (3x4)
     *     length. Balance is in range of [0, 1].
     */
    public static void fisheye_estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat P) {
        fisheye_estimateNewCameraMatrixForUndistortRectify_3(K.nativeObj, D.nativeObj, image_size.width, image_size.height, R.nativeObj, P.nativeObj);
    }


    //
    // C++:  void cv::fisheye::initUndistortRectifyMap(Mat K, Mat D, Mat R, Mat P, Size size, int m1type, Mat& map1, Mat& map2)
    //

    /**
     * Computes undistortion and rectification maps for image transform by cv::remap(). If D is empty zero
     *     distortion is used, if R or P is empty identity matrixes are used.
     *
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
     *     1-channel or 1x1 3-channel
     *     @param P New camera matrix (3x3) or new projection matrix (3x4)
     *     @param size Undistorted image size.
     *     @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See convertMaps()
     *     for details.
     *     @param map1 The first output map.
     *     @param map2 The second output map.
     */
    public static void fisheye_initUndistortRectifyMap(Mat K, Mat D, Mat R, Mat P, Size size, int m1type, Mat map1, Mat map2) {
        fisheye_initUndistortRectifyMap_0(K.nativeObj, D.nativeObj, R.nativeObj, P.nativeObj, size.width, size.height, m1type, map1.nativeObj, map2.nativeObj);
    }


    //
    // C++:  void cv::fisheye::projectPoints(Mat objectPoints, Mat& imagePoints, Mat rvec, Mat tvec, Mat K, Mat D, double alpha = 0, Mat& jacobian = Mat())
    //

    public static void fisheye_projectPoints(Mat objectPoints, Mat imagePoints, Mat rvec, Mat tvec, Mat K, Mat D, double alpha, Mat jacobian) {
        fisheye_projectPoints_0(objectPoints.nativeObj, imagePoints.nativeObj, rvec.nativeObj, tvec.nativeObj, K.nativeObj, D.nativeObj, alpha, jacobian.nativeObj);
    }

    public static void fisheye_projectPoints(Mat objectPoints, Mat imagePoints, Mat rvec, Mat tvec, Mat K, Mat D, double alpha) {
        fisheye_projectPoints_1(objectPoints.nativeObj, imagePoints.nativeObj, rvec.nativeObj, tvec.nativeObj, K.nativeObj, D.nativeObj, alpha);
    }

    public static void fisheye_projectPoints(Mat objectPoints, Mat imagePoints, Mat rvec, Mat tvec, Mat K, Mat D) {
        fisheye_projectPoints_2(objectPoints.nativeObj, imagePoints.nativeObj, rvec.nativeObj, tvec.nativeObj, K.nativeObj, D.nativeObj);
    }


    //
    // C++:  void cv::fisheye::stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, int flags, Size newImageSize = Size(), double balance = 0.0, double fov_scale = 1.0)
    //

    /**
     * Stereo rectification for fisheye camera model
     *
     *     @param K1 First camera matrix.
     *     @param D1 First camera distortion parameters.
     *     @param K2 Second camera matrix.
     *     @param D2 Second camera distortion parameters.
     *     @param imageSize Size of the image used for stereo calibration.
     *     @param R Rotation matrix between the coordinate systems of the first and the second
     *     cameras.
     *     @param tvec Translation vector between coordinate systems of the cameras.
     *     @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
     *     @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
     *     @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     *     camera.
     *     @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     *     camera.
     *     @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see reprojectImageTo3D ).
     *     @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     *     the function makes the principal points of each camera have the same pixel coordinates in the
     *     rectified views. And if the flag is not set, the function may still shift the images in the
     *     horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     *     useful image area.
     *     @param newImageSize New image resolution after rectification. The same size should be passed to
     *     initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     *     is passed (default), it is set to the original imageSize . Setting it to larger value can help you
     *     preserve details in the original image, especially when there is a big radial distortion.
     *     @param balance Sets the new focal length in range between the min focal length and the max focal
     *     length. Balance is in range of [0, 1].
     *     @param fov_scale Divisor for new focal length.
     */
    public static void fisheye_stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, Size newImageSize, double balance, double fov_scale) {
        fisheye_stereoRectify_0(K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, tvec.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, newImageSize.width, newImageSize.height, balance, fov_scale);
    }

    /**
     * Stereo rectification for fisheye camera model
     *
     *     @param K1 First camera matrix.
     *     @param D1 First camera distortion parameters.
     *     @param K2 Second camera matrix.
     *     @param D2 Second camera distortion parameters.
     *     @param imageSize Size of the image used for stereo calibration.
     *     @param R Rotation matrix between the coordinate systems of the first and the second
     *     cameras.
     *     @param tvec Translation vector between coordinate systems of the cameras.
     *     @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
     *     @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
     *     @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     *     camera.
     *     @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     *     camera.
     *     @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see reprojectImageTo3D ).
     *     @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     *     the function makes the principal points of each camera have the same pixel coordinates in the
     *     rectified views. And if the flag is not set, the function may still shift the images in the
     *     horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     *     useful image area.
     *     @param newImageSize New image resolution after rectification. The same size should be passed to
     *     initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     *     is passed (default), it is set to the original imageSize . Setting it to larger value can help you
     *     preserve details in the original image, especially when there is a big radial distortion.
     *     @param balance Sets the new focal length in range between the min focal length and the max focal
     *     length. Balance is in range of [0, 1].
     */
    public static void fisheye_stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, Size newImageSize, double balance) {
        fisheye_stereoRectify_1(K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, tvec.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, newImageSize.width, newImageSize.height, balance);
    }

    /**
     * Stereo rectification for fisheye camera model
     *
     *     @param K1 First camera matrix.
     *     @param D1 First camera distortion parameters.
     *     @param K2 Second camera matrix.
     *     @param D2 Second camera distortion parameters.
     *     @param imageSize Size of the image used for stereo calibration.
     *     @param R Rotation matrix between the coordinate systems of the first and the second
     *     cameras.
     *     @param tvec Translation vector between coordinate systems of the cameras.
     *     @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
     *     @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
     *     @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     *     camera.
     *     @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     *     camera.
     *     @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see reprojectImageTo3D ).
     *     @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     *     the function makes the principal points of each camera have the same pixel coordinates in the
     *     rectified views. And if the flag is not set, the function may still shift the images in the
     *     horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     *     useful image area.
     *     @param newImageSize New image resolution after rectification. The same size should be passed to
     *     initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     *     is passed (default), it is set to the original imageSize . Setting it to larger value can help you
     *     preserve details in the original image, especially when there is a big radial distortion.
     *     length. Balance is in range of [0, 1].
     */
    public static void fisheye_stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, Size newImageSize) {
        fisheye_stereoRectify_2(K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, tvec.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, newImageSize.width, newImageSize.height);
    }

    /**
     * Stereo rectification for fisheye camera model
     *
     *     @param K1 First camera matrix.
     *     @param D1 First camera distortion parameters.
     *     @param K2 Second camera matrix.
     *     @param D2 Second camera distortion parameters.
     *     @param imageSize Size of the image used for stereo calibration.
     *     @param R Rotation matrix between the coordinate systems of the first and the second
     *     cameras.
     *     @param tvec Translation vector between coordinate systems of the cameras.
     *     @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
     *     @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
     *     @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
     *     camera.
     *     @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
     *     camera.
     *     @param Q Output \(4 \times 4\) disparity-to-depth mapping matrix (see reprojectImageTo3D ).
     *     @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
     *     the function makes the principal points of each camera have the same pixel coordinates in the
     *     rectified views. And if the flag is not set, the function may still shift the images in the
     *     horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
     *     useful image area.
     *     initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
     *     is passed (default), it is set to the original imageSize . Setting it to larger value can help you
     *     preserve details in the original image, especially when there is a big radial distortion.
     *     length. Balance is in range of [0, 1].
     */
    public static void fisheye_stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags) {
        fisheye_stereoRectify_3(K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, tvec.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags);
    }


    //
    // C++:  void cv::fisheye::undistortImage(Mat distorted, Mat& undistorted, Mat K, Mat D, Mat Knew = cv::Mat(), Size new_size = Size())
    //

    /**
     * Transforms an image to compensate for fisheye lens distortion.
     *
     *     @param distorted image with fisheye lens distortion.
     *     @param undistorted Output image with compensated fisheye lens distortion.
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param Knew Camera matrix of the distorted image. By default, it is the identity matrix but you
     *     may additionally scale and shift the result by using a different matrix.
     *     @param new_size the new size
     *
     *     The function transforms an image to compensate radial and tangential lens distortion.
     *
     *     The function is simply a combination of fisheye::initUndistortRectifyMap (with unity R ) and remap
     *     (with bilinear interpolation). See the former function for details of the transformation being
     *     performed.
     *
     *     See below the results of undistortImage.
     * <ul>
     *   <li>
     *           a\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
     *             k_4, k_5, k_6) of distortion were optimized under calibration)
     *   <ul>
     *     <li>
     *            b\) result of fisheye::undistortImage of fisheye camera model (all possible coefficients (k_1, k_2,
     *             k_3, k_4) of fisheye distortion were optimized under calibration)
     *     </li>
     *     <li>
     *            c\) original image was captured with fisheye lens
     *     </li>
     *   </ul>
     *
     *     Pictures a) and b) almost the same. But if we consider points of image located far from the center
     *     of image, we can notice that on image a) these points are distorted.
     *   </li>
     * </ul>
     *
     *     ![image](pics/fisheye_undistorted.jpg)
     */
    public static void fisheye_undistortImage(Mat distorted, Mat undistorted, Mat K, Mat D, Mat Knew, Size new_size) {
        fisheye_undistortImage_0(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj, Knew.nativeObj, new_size.width, new_size.height);
    }

    /**
     * Transforms an image to compensate for fisheye lens distortion.
     *
     *     @param distorted image with fisheye lens distortion.
     *     @param undistorted Output image with compensated fisheye lens distortion.
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param Knew Camera matrix of the distorted image. By default, it is the identity matrix but you
     *     may additionally scale and shift the result by using a different matrix.
     *
     *     The function transforms an image to compensate radial and tangential lens distortion.
     *
     *     The function is simply a combination of fisheye::initUndistortRectifyMap (with unity R ) and remap
     *     (with bilinear interpolation). See the former function for details of the transformation being
     *     performed.
     *
     *     See below the results of undistortImage.
     * <ul>
     *   <li>
     *           a\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
     *             k_4, k_5, k_6) of distortion were optimized under calibration)
     *   <ul>
     *     <li>
     *            b\) result of fisheye::undistortImage of fisheye camera model (all possible coefficients (k_1, k_2,
     *             k_3, k_4) of fisheye distortion were optimized under calibration)
     *     </li>
     *     <li>
     *            c\) original image was captured with fisheye lens
     *     </li>
     *   </ul>
     *
     *     Pictures a) and b) almost the same. But if we consider points of image located far from the center
     *     of image, we can notice that on image a) these points are distorted.
     *   </li>
     * </ul>
     *
     *     ![image](pics/fisheye_undistorted.jpg)
     */
    public static void fisheye_undistortImage(Mat distorted, Mat undistorted, Mat K, Mat D, Mat Knew) {
        fisheye_undistortImage_1(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj, Knew.nativeObj);
    }

    /**
     * Transforms an image to compensate for fisheye lens distortion.
     *
     *     @param distorted image with fisheye lens distortion.
     *     @param undistorted Output image with compensated fisheye lens distortion.
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     may additionally scale and shift the result by using a different matrix.
     *
     *     The function transforms an image to compensate radial and tangential lens distortion.
     *
     *     The function is simply a combination of fisheye::initUndistortRectifyMap (with unity R ) and remap
     *     (with bilinear interpolation). See the former function for details of the transformation being
     *     performed.
     *
     *     See below the results of undistortImage.
     * <ul>
     *   <li>
     *           a\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
     *             k_4, k_5, k_6) of distortion were optimized under calibration)
     *   <ul>
     *     <li>
     *            b\) result of fisheye::undistortImage of fisheye camera model (all possible coefficients (k_1, k_2,
     *             k_3, k_4) of fisheye distortion were optimized under calibration)
     *     </li>
     *     <li>
     *            c\) original image was captured with fisheye lens
     *     </li>
     *   </ul>
     *
     *     Pictures a) and b) almost the same. But if we consider points of image located far from the center
     *     of image, we can notice that on image a) these points are distorted.
     *   </li>
     * </ul>
     *
     *     ![image](pics/fisheye_undistorted.jpg)
     */
    public static void fisheye_undistortImage(Mat distorted, Mat undistorted, Mat K, Mat D) {
        fisheye_undistortImage_2(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj);
    }


    //
    // C++:  void cv::fisheye::undistortPoints(Mat distorted, Mat& undistorted, Mat K, Mat D, Mat R = Mat(), Mat P = Mat())
    //

    /**
     * Undistorts 2D points using fisheye model
     *
     *     @param distorted Array of object points, 1xN/Nx1 2-channel (or vector&lt;Point2f&gt; ), where N is the
     *     number of points in the view.
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
     *     1-channel or 1x1 3-channel
     *     @param P New camera matrix (3x3) or new projection matrix (3x4)
     *     @param undistorted Output array of image points, 1xN/Nx1 2-channel, or vector&lt;Point2f&gt; .
     */
    public static void fisheye_undistortPoints(Mat distorted, Mat undistorted, Mat K, Mat D, Mat R, Mat P) {
        fisheye_undistortPoints_0(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj, R.nativeObj, P.nativeObj);
    }

    /**
     * Undistorts 2D points using fisheye model
     *
     *     @param distorted Array of object points, 1xN/Nx1 2-channel (or vector&lt;Point2f&gt; ), where N is the
     *     number of points in the view.
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
     *     1-channel or 1x1 3-channel
     *     @param undistorted Output array of image points, 1xN/Nx1 2-channel, or vector&lt;Point2f&gt; .
     */
    public static void fisheye_undistortPoints(Mat distorted, Mat undistorted, Mat K, Mat D, Mat R) {
        fisheye_undistortPoints_1(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj, R.nativeObj);
    }

    /**
     * Undistorts 2D points using fisheye model
     *
     *     @param distorted Array of object points, 1xN/Nx1 2-channel (or vector&lt;Point2f&gt; ), where N is the
     *     number of points in the view.
     *     @param K Camera matrix \(K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\).
     *     @param D Input vector of distortion coefficients \((k_1, k_2, k_3, k_4)\).
     *     1-channel or 1x1 3-channel
     *     @param undistorted Output array of image points, 1xN/Nx1 2-channel, or vector&lt;Point2f&gt; .
     */
    public static void fisheye_undistortPoints(Mat distorted, Mat undistorted, Mat K, Mat D) {
        fisheye_undistortPoints_2(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj);
    }




    // C++:  Mat cv::estimateAffine2D(Mat from, Mat to, Mat& inliers = Mat(), int method = RANSAC, double ransacReprojThreshold = 3, size_t maxIters = 2000, double confidence = 0.99, size_t refineIters = 10)
    private static native long estimateAffine2D_0(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method, double ransacReprojThreshold, long maxIters, double confidence, long refineIters);
    private static native long estimateAffine2D_1(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method, double ransacReprojThreshold, long maxIters, double confidence);
    private static native long estimateAffine2D_2(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method, double ransacReprojThreshold, long maxIters);
    private static native long estimateAffine2D_3(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method, double ransacReprojThreshold);
    private static native long estimateAffine2D_4(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method);
    private static native long estimateAffine2D_5(long from_nativeObj, long to_nativeObj, long inliers_nativeObj);
    private static native long estimateAffine2D_6(long from_nativeObj, long to_nativeObj);

    // C++:  Mat cv::estimateAffinePartial2D(Mat from, Mat to, Mat& inliers = Mat(), int method = RANSAC, double ransacReprojThreshold = 3, size_t maxIters = 2000, double confidence = 0.99, size_t refineIters = 10)
    private static native long estimateAffinePartial2D_0(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method, double ransacReprojThreshold, long maxIters, double confidence, long refineIters);
    private static native long estimateAffinePartial2D_1(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method, double ransacReprojThreshold, long maxIters, double confidence);
    private static native long estimateAffinePartial2D_2(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method, double ransacReprojThreshold, long maxIters);
    private static native long estimateAffinePartial2D_3(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method, double ransacReprojThreshold);
    private static native long estimateAffinePartial2D_4(long from_nativeObj, long to_nativeObj, long inliers_nativeObj, int method);
    private static native long estimateAffinePartial2D_5(long from_nativeObj, long to_nativeObj, long inliers_nativeObj);
    private static native long estimateAffinePartial2D_6(long from_nativeObj, long to_nativeObj);

    // C++:  Mat cv::findEssentialMat(Mat points1, Mat points2, Mat cameraMatrix, int method = RANSAC, double prob = 0.999, double threshold = 1.0, Mat& mask = Mat())
    private static native long findEssentialMat_0(long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, int method, double prob, double threshold, long mask_nativeObj);
    private static native long findEssentialMat_1(long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, int method, double prob, double threshold);
    private static native long findEssentialMat_2(long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, int method, double prob);
    private static native long findEssentialMat_3(long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, int method);
    private static native long findEssentialMat_4(long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj);

    // C++:  Mat cv::findEssentialMat(Mat points1, Mat points2, double focal = 1.0, Point2d pp = Point2d(0, 0), int method = RANSAC, double prob = 0.999, double threshold = 1.0, Mat& mask = Mat())
    private static native long findEssentialMat_5(long points1_nativeObj, long points2_nativeObj, double focal, double pp_x, double pp_y, int method, double prob, double threshold, long mask_nativeObj);
    private static native long findEssentialMat_6(long points1_nativeObj, long points2_nativeObj, double focal, double pp_x, double pp_y, int method, double prob, double threshold);
    private static native long findEssentialMat_7(long points1_nativeObj, long points2_nativeObj, double focal, double pp_x, double pp_y, int method, double prob);
    private static native long findEssentialMat_8(long points1_nativeObj, long points2_nativeObj, double focal, double pp_x, double pp_y, int method);
    private static native long findEssentialMat_9(long points1_nativeObj, long points2_nativeObj, double focal, double pp_x, double pp_y);
    private static native long findEssentialMat_10(long points1_nativeObj, long points2_nativeObj, double focal);
    private static native long findEssentialMat_11(long points1_nativeObj, long points2_nativeObj);

    // C++:  Mat cv::findFundamentalMat(vector_Point2f points1, vector_Point2f points2, int method, double ransacReprojThreshold, double confidence, int maxIters, Mat& mask = Mat())
    private static native long findFundamentalMat_0(long points1_mat_nativeObj, long points2_mat_nativeObj, int method, double ransacReprojThreshold, double confidence, int maxIters, long mask_nativeObj);
    private static native long findFundamentalMat_1(long points1_mat_nativeObj, long points2_mat_nativeObj, int method, double ransacReprojThreshold, double confidence, int maxIters);

    // C++:  Mat cv::findFundamentalMat(vector_Point2f points1, vector_Point2f points2, int method = FM_RANSAC, double ransacReprojThreshold = 3., double confidence = 0.99, Mat& mask = Mat())
    private static native long findFundamentalMat_2(long points1_mat_nativeObj, long points2_mat_nativeObj, int method, double ransacReprojThreshold, double confidence, long mask_nativeObj);
    private static native long findFundamentalMat_3(long points1_mat_nativeObj, long points2_mat_nativeObj, int method, double ransacReprojThreshold, double confidence);
    private static native long findFundamentalMat_4(long points1_mat_nativeObj, long points2_mat_nativeObj, int method, double ransacReprojThreshold);
    private static native long findFundamentalMat_5(long points1_mat_nativeObj, long points2_mat_nativeObj, int method);
    private static native long findFundamentalMat_6(long points1_mat_nativeObj, long points2_mat_nativeObj);

    // C++:  Mat cv::findHomography(vector_Point2f srcPoints, vector_Point2f dstPoints, int method = 0, double ransacReprojThreshold = 3, Mat& mask = Mat(), int maxIters = 2000, double confidence = 0.995)
    private static native long findHomography_0(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj, int method, double ransacReprojThreshold, long mask_nativeObj, int maxIters, double confidence);
    private static native long findHomography_1(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj, int method, double ransacReprojThreshold, long mask_nativeObj, int maxIters);
    private static native long findHomography_2(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj, int method, double ransacReprojThreshold, long mask_nativeObj);
    private static native long findHomography_3(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj, int method, double ransacReprojThreshold);
    private static native long findHomography_4(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj, int method);
    private static native long findHomography_5(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj);

    // C++:  Mat cv::getDefaultNewCameraMatrix(Mat cameraMatrix, Size imgsize = Size(), bool centerPrincipalPoint = false)
    private static native long getDefaultNewCameraMatrix_0(long cameraMatrix_nativeObj, double imgsize_width, double imgsize_height, boolean centerPrincipalPoint);
    private static native long getDefaultNewCameraMatrix_1(long cameraMatrix_nativeObj, double imgsize_width, double imgsize_height);
    private static native long getDefaultNewCameraMatrix_2(long cameraMatrix_nativeObj);

    // C++:  Mat cv::getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha, Size newImgSize = Size(), Rect* validPixROI = 0, bool centerPrincipalPoint = false)
    private static native long getOptimalNewCameraMatrix_0(long cameraMatrix_nativeObj, long distCoeffs_nativeObj, double imageSize_width, double imageSize_height, double alpha, double newImgSize_width, double newImgSize_height, double[] validPixROI_out, boolean centerPrincipalPoint);
    private static native long getOptimalNewCameraMatrix_1(long cameraMatrix_nativeObj, long distCoeffs_nativeObj, double imageSize_width, double imageSize_height, double alpha, double newImgSize_width, double newImgSize_height, double[] validPixROI_out);
    private static native long getOptimalNewCameraMatrix_2(long cameraMatrix_nativeObj, long distCoeffs_nativeObj, double imageSize_width, double imageSize_height, double alpha, double newImgSize_width, double newImgSize_height);
    private static native long getOptimalNewCameraMatrix_3(long cameraMatrix_nativeObj, long distCoeffs_nativeObj, double imageSize_width, double imageSize_height, double alpha);

    // C++:  Mat cv::initCameraMatrix2D(vector_vector_Point3f objectPoints, vector_vector_Point2f imagePoints, Size imageSize, double aspectRatio = 1.0)
    private static native long initCameraMatrix2D_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, double aspectRatio);
    private static native long initCameraMatrix2D_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height);

    // C++:  Rect cv::getValidDisparityROI(Rect roi1, Rect roi2, int minDisparity, int numberOfDisparities, int blockSize)
    private static native double[] getValidDisparityROI_0(int roi1_x, int roi1_y, int roi1_width, int roi1_height, int roi2_x, int roi2_y, int roi2_width, int roi2_height, int minDisparity, int numberOfDisparities, int blockSize);

    // C++:  Scalar cv::estimateChessboardSharpness(Mat image, Size patternSize, Mat corners, float rise_distance = 0.8F, bool vertical = false, Mat& sharpness = Mat())
    private static native double[] estimateChessboardSharpness_0(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_nativeObj, float rise_distance, boolean vertical, long sharpness_nativeObj);
    private static native double[] estimateChessboardSharpness_1(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_nativeObj, float rise_distance, boolean vertical);
    private static native double[] estimateChessboardSharpness_2(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_nativeObj, float rise_distance);
    private static native double[] estimateChessboardSharpness_3(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_nativeObj);

    // C++:  Vec3d cv::RQDecomp3x3(Mat src, Mat& mtxR, Mat& mtxQ, Mat& Qx = Mat(), Mat& Qy = Mat(), Mat& Qz = Mat())
    private static native double[] RQDecomp3x3_0(long src_nativeObj, long mtxR_nativeObj, long mtxQ_nativeObj, long Qx_nativeObj, long Qy_nativeObj, long Qz_nativeObj);
    private static native double[] RQDecomp3x3_1(long src_nativeObj, long mtxR_nativeObj, long mtxQ_nativeObj, long Qx_nativeObj, long Qy_nativeObj);
    private static native double[] RQDecomp3x3_2(long src_nativeObj, long mtxR_nativeObj, long mtxQ_nativeObj, long Qx_nativeObj);
    private static native double[] RQDecomp3x3_3(long src_nativeObj, long mtxR_nativeObj, long mtxQ_nativeObj);

    // C++:  bool cv::checkChessboard(Mat img, Size size)
    private static native boolean checkChessboard_0(long img_nativeObj, double size_width, double size_height);

    // C++:  bool cv::find4QuadCornerSubpix(Mat img, Mat& corners, Size region_size)
    private static native boolean find4QuadCornerSubpix_0(long img_nativeObj, long corners_nativeObj, double region_size_width, double region_size_height);

    // C++:  bool cv::findChessboardCorners(Mat image, Size patternSize, vector_Point2f& corners, int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE)
    private static native boolean findChessboardCorners_0(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_mat_nativeObj, int flags);
    private static native boolean findChessboardCorners_1(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_mat_nativeObj);

    // C++:  bool cv::findChessboardCornersSB(Mat image, Size patternSize, Mat& corners, int flags, Mat& meta)
    private static native boolean findChessboardCornersSBWithMeta_0(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_nativeObj, int flags, long meta_nativeObj);

    // C++:  bool cv::findChessboardCornersSB(Mat image, Size patternSize, Mat& corners, int flags = 0)
    private static native boolean findChessboardCornersSB_0(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_nativeObj, int flags);
    private static native boolean findChessboardCornersSB_1(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_nativeObj);

    // C++:  bool cv::findCirclesGrid(Mat image, Size patternSize, Mat& centers, int flags = CALIB_CB_SYMMETRIC_GRID, Ptr_FeatureDetector blobDetector = SimpleBlobDetector::create())
    private static native boolean findCirclesGrid_0(long image_nativeObj, double patternSize_width, double patternSize_height, long centers_nativeObj, int flags);
    private static native boolean findCirclesGrid_2(long image_nativeObj, double patternSize_width, double patternSize_height, long centers_nativeObj);

    // C++:  bool cv::solvePnP(vector_Point3f objectPoints, vector_Point2f imagePoints, Mat cameraMatrix, vector_double distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE)
    private static native boolean solvePnP_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess, int flags);
    private static native boolean solvePnP_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess);
    private static native boolean solvePnP_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj);

    // C++:  bool cv::solvePnPRansac(vector_Point3f objectPoints, vector_Point2f imagePoints, Mat cameraMatrix, vector_double distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess = false, int iterationsCount = 100, float reprojectionError = 8.0, double confidence = 0.99, Mat& inliers = Mat(), int flags = SOLVEPNP_ITERATIVE)
    private static native boolean solvePnPRansac_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError, double confidence, long inliers_nativeObj, int flags);
    private static native boolean solvePnPRansac_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError, double confidence, long inliers_nativeObj);
    private static native boolean solvePnPRansac_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError, double confidence);
    private static native boolean solvePnPRansac_3(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError);
    private static native boolean solvePnPRansac_4(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess, int iterationsCount);
    private static native boolean solvePnPRansac_5(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess);
    private static native boolean solvePnPRansac_6(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj);

    // C++:  bool cv::stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, Mat& H1, Mat& H2, double threshold = 5)
    private static native boolean stereoRectifyUncalibrated_0(long points1_nativeObj, long points2_nativeObj, long F_nativeObj, double imgSize_width, double imgSize_height, long H1_nativeObj, long H2_nativeObj, double threshold);
    private static native boolean stereoRectifyUncalibrated_1(long points1_nativeObj, long points2_nativeObj, long F_nativeObj, double imgSize_width, double imgSize_height, long H1_nativeObj, long H2_nativeObj);

    // C++:  double cv::calibrateCamera(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, Mat& stdDeviationsIntrinsics, Mat& stdDeviationsExtrinsics, Mat& perViewErrors, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    private static native double calibrateCameraExtended_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long stdDeviationsIntrinsics_nativeObj, long stdDeviationsExtrinsics_nativeObj, long perViewErrors_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double calibrateCameraExtended_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long stdDeviationsIntrinsics_nativeObj, long stdDeviationsExtrinsics_nativeObj, long perViewErrors_nativeObj, int flags);
    private static native double calibrateCameraExtended_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long stdDeviationsIntrinsics_nativeObj, long stdDeviationsExtrinsics_nativeObj, long perViewErrors_nativeObj);

    // C++:  double cv::calibrateCamera(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    private static native double calibrateCamera_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double calibrateCamera_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags);
    private static native double calibrateCamera_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj);

    // C++:  double cv::calibrateCameraRO(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, int iFixedPoint, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, Mat& newObjPoints, Mat& stdDeviationsIntrinsics, Mat& stdDeviationsExtrinsics, Mat& stdDeviationsObjPoints, Mat& perViewErrors, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    private static native double calibrateCameraROExtended_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, int iFixedPoint, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long newObjPoints_nativeObj, long stdDeviationsIntrinsics_nativeObj, long stdDeviationsExtrinsics_nativeObj, long stdDeviationsObjPoints_nativeObj, long perViewErrors_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double calibrateCameraROExtended_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, int iFixedPoint, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long newObjPoints_nativeObj, long stdDeviationsIntrinsics_nativeObj, long stdDeviationsExtrinsics_nativeObj, long stdDeviationsObjPoints_nativeObj, long perViewErrors_nativeObj, int flags);
    private static native double calibrateCameraROExtended_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, int iFixedPoint, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long newObjPoints_nativeObj, long stdDeviationsIntrinsics_nativeObj, long stdDeviationsExtrinsics_nativeObj, long stdDeviationsObjPoints_nativeObj, long perViewErrors_nativeObj);

    // C++:  double cv::calibrateCameraRO(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, int iFixedPoint, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, Mat& newObjPoints, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    private static native double calibrateCameraRO_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, int iFixedPoint, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long newObjPoints_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double calibrateCameraRO_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, int iFixedPoint, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long newObjPoints_nativeObj, int flags);
    private static native double calibrateCameraRO_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, int iFixedPoint, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, long newObjPoints_nativeObj);

    // C++:  double cv::sampsonDistance(Mat pt1, Mat pt2, Mat F)
    private static native double sampsonDistance_0(long pt1_nativeObj, long pt2_nativeObj, long F_nativeObj);

    // C++:  double cv::stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size imageSize, Mat& R, Mat& T, Mat& E, Mat& F, Mat& perViewErrors, int flags = CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))
    private static native double stereoCalibrateExtended_0(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj, long perViewErrors_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double stereoCalibrateExtended_1(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj, long perViewErrors_nativeObj, int flags);
    private static native double stereoCalibrateExtended_2(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj, long perViewErrors_nativeObj);

    // C++:  double cv::stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size imageSize, Mat& R, Mat& T, Mat& E, Mat& F, int flags = CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))
    private static native double stereoCalibrate_0(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double stereoCalibrate_1(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj, int flags);
    private static native double stereoCalibrate_2(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj);

    // C++:  double cv::fisheye::calibrate(vector_Mat objectPoints, vector_Mat imagePoints, Size image_size, Mat& K, Mat& D, vector_Mat& rvecs, vector_Mat& tvecs, int flags = 0, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON))
    private static native double fisheye_calibrate_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double image_size_width, double image_size_height, long K_nativeObj, long D_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double fisheye_calibrate_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double image_size_width, double image_size_height, long K_nativeObj, long D_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags);
    private static native double fisheye_calibrate_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double image_size_width, double image_size_height, long K_nativeObj, long D_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj);

    // C++:  double cv::fisheye::stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& K1, Mat& D1, Mat& K2, Mat& D2, Size imageSize, Mat& R, Mat& T, int flags = fisheye::CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON))
    private static native double fisheye_stereoCalibrate_0(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double fisheye_stereoCalibrate_1(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, int flags);
    private static native double fisheye_stereoCalibrate_2(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj);

    // C++:  float cv::rectify3Collinear(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Mat cameraMatrix3, Mat distCoeffs3, vector_Mat imgpt1, vector_Mat imgpt3, Size imageSize, Mat R12, Mat T12, Mat R13, Mat T13, Mat& R1, Mat& R2, Mat& R3, Mat& P1, Mat& P2, Mat& P3, Mat& Q, double alpha, Size newImgSize, Rect* roi1, Rect* roi2, int flags)
    private static native float rectify3Collinear_0(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, long cameraMatrix3_nativeObj, long distCoeffs3_nativeObj, long imgpt1_mat_nativeObj, long imgpt3_mat_nativeObj, double imageSize_width, double imageSize_height, long R12_nativeObj, long T12_nativeObj, long R13_nativeObj, long T13_nativeObj, long R1_nativeObj, long R2_nativeObj, long R3_nativeObj, long P1_nativeObj, long P2_nativeObj, long P3_nativeObj, long Q_nativeObj, double alpha, double newImgSize_width, double newImgSize_height, double[] roi1_out, double[] roi2_out, int flags);

    // C++:  int cv::decomposeHomographyMat(Mat H, Mat K, vector_Mat& rotations, vector_Mat& translations, vector_Mat& normals)
    private static native int decomposeHomographyMat_0(long H_nativeObj, long K_nativeObj, long rotations_mat_nativeObj, long translations_mat_nativeObj, long normals_mat_nativeObj);

    // C++:  int cv::estimateAffine3D(Mat src, Mat dst, Mat& out, Mat& inliers, double ransacThreshold = 3, double confidence = 0.99)
    private static native int estimateAffine3D_0(long src_nativeObj, long dst_nativeObj, long out_nativeObj, long inliers_nativeObj, double ransacThreshold, double confidence);
    private static native int estimateAffine3D_1(long src_nativeObj, long dst_nativeObj, long out_nativeObj, long inliers_nativeObj, double ransacThreshold);
    private static native int estimateAffine3D_2(long src_nativeObj, long dst_nativeObj, long out_nativeObj, long inliers_nativeObj);

    // C++:  int cv::recoverPose(Mat E, Mat points1, Mat points2, Mat& R, Mat& t, double focal = 1.0, Point2d pp = Point2d(0, 0), Mat& mask = Mat())
    private static native int recoverPose_0(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long R_nativeObj, long t_nativeObj, double focal, double pp_x, double pp_y, long mask_nativeObj);
    private static native int recoverPose_1(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long R_nativeObj, long t_nativeObj, double focal, double pp_x, double pp_y);
    private static native int recoverPose_2(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long R_nativeObj, long t_nativeObj, double focal);
    private static native int recoverPose_3(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long R_nativeObj, long t_nativeObj);

    // C++:  int cv::recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat& R, Mat& t, Mat& mask = Mat())
    private static native int recoverPose_4(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, long R_nativeObj, long t_nativeObj, long mask_nativeObj);
    private static native int recoverPose_5(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, long R_nativeObj, long t_nativeObj);

    // C++:  int cv::recoverPose(Mat E, Mat points1, Mat points2, Mat cameraMatrix, Mat& R, Mat& t, double distanceThresh, Mat& mask = Mat(), Mat& triangulatedPoints = Mat())
    private static native int recoverPose_6(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, long R_nativeObj, long t_nativeObj, double distanceThresh, long mask_nativeObj, long triangulatedPoints_nativeObj);
    private static native int recoverPose_7(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, long R_nativeObj, long t_nativeObj, double distanceThresh, long mask_nativeObj);
    private static native int recoverPose_8(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long cameraMatrix_nativeObj, long R_nativeObj, long t_nativeObj, double distanceThresh);

    // C++:  int cv::solveP3P(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, int flags)
    private static native int solveP3P_0(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags);

    // C++:  int cv::solvePnPGeneric(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, bool useExtrinsicGuess = false, SolvePnPMethod flags = SOLVEPNP_ITERATIVE, Mat rvec = Mat(), Mat tvec = Mat(), Mat& reprojectionError = Mat())
    private static native int solvePnPGeneric_0(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, boolean useExtrinsicGuess, int flags, long rvec_nativeObj, long tvec_nativeObj, long reprojectionError_nativeObj);
    private static native int solvePnPGeneric_1(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, boolean useExtrinsicGuess, int flags, long rvec_nativeObj, long tvec_nativeObj);
    private static native int solvePnPGeneric_2(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, boolean useExtrinsicGuess, int flags, long rvec_nativeObj);
    private static native int solvePnPGeneric_3(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, boolean useExtrinsicGuess, int flags);
    private static native int solvePnPGeneric_4(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, boolean useExtrinsicGuess);
    private static native int solvePnPGeneric_5(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj);

    // C++:  void cv::Rodrigues(Mat src, Mat& dst, Mat& jacobian = Mat())
    private static native void Rodrigues_0(long src_nativeObj, long dst_nativeObj, long jacobian_nativeObj);
    private static native void Rodrigues_1(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::calibrateHandEye(vector_Mat R_gripper2base, vector_Mat t_gripper2base, vector_Mat R_target2cam, vector_Mat t_target2cam, Mat& R_cam2gripper, Mat& t_cam2gripper, HandEyeCalibrationMethod method = CALIB_HAND_EYE_TSAI)
    private static native void calibrateHandEye_0(long R_gripper2base_mat_nativeObj, long t_gripper2base_mat_nativeObj, long R_target2cam_mat_nativeObj, long t_target2cam_mat_nativeObj, long R_cam2gripper_nativeObj, long t_cam2gripper_nativeObj, int method);
    private static native void calibrateHandEye_1(long R_gripper2base_mat_nativeObj, long t_gripper2base_mat_nativeObj, long R_target2cam_mat_nativeObj, long t_target2cam_mat_nativeObj, long R_cam2gripper_nativeObj, long t_cam2gripper_nativeObj);

    // C++:  void cv::calibrationMatrixValues(Mat cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength, Point2d& principalPoint, double& aspectRatio)
    private static native void calibrationMatrixValues_0(long cameraMatrix_nativeObj, double imageSize_width, double imageSize_height, double apertureWidth, double apertureHeight, double[] fovx_out, double[] fovy_out, double[] focalLength_out, double[] principalPoint_out, double[] aspectRatio_out);

    // C++:  void cv::composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat& rvec3, Mat& tvec3, Mat& dr3dr1 = Mat(), Mat& dr3dt1 = Mat(), Mat& dr3dr2 = Mat(), Mat& dr3dt2 = Mat(), Mat& dt3dr1 = Mat(), Mat& dt3dt1 = Mat(), Mat& dt3dr2 = Mat(), Mat& dt3dt2 = Mat())
    private static native void composeRT_0(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj, long dr3dt1_nativeObj, long dr3dr2_nativeObj, long dr3dt2_nativeObj, long dt3dr1_nativeObj, long dt3dt1_nativeObj, long dt3dr2_nativeObj, long dt3dt2_nativeObj);
    private static native void composeRT_1(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj, long dr3dt1_nativeObj, long dr3dr2_nativeObj, long dr3dt2_nativeObj, long dt3dr1_nativeObj, long dt3dt1_nativeObj, long dt3dr2_nativeObj);
    private static native void composeRT_2(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj, long dr3dt1_nativeObj, long dr3dr2_nativeObj, long dr3dt2_nativeObj, long dt3dr1_nativeObj, long dt3dt1_nativeObj);
    private static native void composeRT_3(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj, long dr3dt1_nativeObj, long dr3dr2_nativeObj, long dr3dt2_nativeObj, long dt3dr1_nativeObj);
    private static native void composeRT_4(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj, long dr3dt1_nativeObj, long dr3dr2_nativeObj, long dr3dt2_nativeObj);
    private static native void composeRT_5(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj, long dr3dt1_nativeObj, long dr3dr2_nativeObj);
    private static native void composeRT_6(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj, long dr3dt1_nativeObj);
    private static native void composeRT_7(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj);
    private static native void composeRT_8(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj);

    // C++:  void cv::computeCorrespondEpilines(Mat points, int whichImage, Mat F, Mat& lines)
    private static native void computeCorrespondEpilines_0(long points_nativeObj, int whichImage, long F_nativeObj, long lines_nativeObj);

    // C++:  void cv::convertPointsFromHomogeneous(Mat src, Mat& dst)
    private static native void convertPointsFromHomogeneous_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::convertPointsToHomogeneous(Mat src, Mat& dst)
    private static native void convertPointsToHomogeneous_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::correctMatches(Mat F, Mat points1, Mat points2, Mat& newPoints1, Mat& newPoints2)
    private static native void correctMatches_0(long F_nativeObj, long points1_nativeObj, long points2_nativeObj, long newPoints1_nativeObj, long newPoints2_nativeObj);

    // C++:  void cv::decomposeEssentialMat(Mat E, Mat& R1, Mat& R2, Mat& t)
    private static native void decomposeEssentialMat_0(long E_nativeObj, long R1_nativeObj, long R2_nativeObj, long t_nativeObj);

    // C++:  void cv::decomposeProjectionMatrix(Mat projMatrix, Mat& cameraMatrix, Mat& rotMatrix, Mat& transVect, Mat& rotMatrixX = Mat(), Mat& rotMatrixY = Mat(), Mat& rotMatrixZ = Mat(), Mat& eulerAngles = Mat())
    private static native void decomposeProjectionMatrix_0(long projMatrix_nativeObj, long cameraMatrix_nativeObj, long rotMatrix_nativeObj, long transVect_nativeObj, long rotMatrixX_nativeObj, long rotMatrixY_nativeObj, long rotMatrixZ_nativeObj, long eulerAngles_nativeObj);
    private static native void decomposeProjectionMatrix_1(long projMatrix_nativeObj, long cameraMatrix_nativeObj, long rotMatrix_nativeObj, long transVect_nativeObj, long rotMatrixX_nativeObj, long rotMatrixY_nativeObj, long rotMatrixZ_nativeObj);
    private static native void decomposeProjectionMatrix_2(long projMatrix_nativeObj, long cameraMatrix_nativeObj, long rotMatrix_nativeObj, long transVect_nativeObj, long rotMatrixX_nativeObj, long rotMatrixY_nativeObj);
    private static native void decomposeProjectionMatrix_3(long projMatrix_nativeObj, long cameraMatrix_nativeObj, long rotMatrix_nativeObj, long transVect_nativeObj, long rotMatrixX_nativeObj);
    private static native void decomposeProjectionMatrix_4(long projMatrix_nativeObj, long cameraMatrix_nativeObj, long rotMatrix_nativeObj, long transVect_nativeObj);

    // C++:  void cv::drawChessboardCorners(Mat& image, Size patternSize, vector_Point2f corners, bool patternWasFound)
    private static native void drawChessboardCorners_0(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_mat_nativeObj, boolean patternWasFound);

    // C++:  void cv::drawFrameAxes(Mat& image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, float length, int thickness = 3)
    private static native void drawFrameAxes_0(long image_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvec_nativeObj, long tvec_nativeObj, float length, int thickness);
    private static native void drawFrameAxes_1(long image_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvec_nativeObj, long tvec_nativeObj, float length);

    // C++:  void cv::filterHomographyDecompByVisibleRefpoints(vector_Mat rotations, vector_Mat normals, Mat beforePoints, Mat afterPoints, Mat& possibleSolutions, Mat pointsMask = Mat())
    private static native void filterHomographyDecompByVisibleRefpoints_0(long rotations_mat_nativeObj, long normals_mat_nativeObj, long beforePoints_nativeObj, long afterPoints_nativeObj, long possibleSolutions_nativeObj, long pointsMask_nativeObj);
    private static native void filterHomographyDecompByVisibleRefpoints_1(long rotations_mat_nativeObj, long normals_mat_nativeObj, long beforePoints_nativeObj, long afterPoints_nativeObj, long possibleSolutions_nativeObj);

    // C++:  void cv::filterSpeckles(Mat& img, double newVal, int maxSpeckleSize, double maxDiff, Mat& buf = Mat())
    private static native void filterSpeckles_0(long img_nativeObj, double newVal, int maxSpeckleSize, double maxDiff, long buf_nativeObj);
    private static native void filterSpeckles_1(long img_nativeObj, double newVal, int maxSpeckleSize, double maxDiff);

    // C++:  void cv::initUndistortRectifyMap(Mat cameraMatrix, Mat distCoeffs, Mat R, Mat newCameraMatrix, Size size, int m1type, Mat& map1, Mat& map2)
    private static native void initUndistortRectifyMap_0(long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long R_nativeObj, long newCameraMatrix_nativeObj, double size_width, double size_height, int m1type, long map1_nativeObj, long map2_nativeObj);

    // C++:  void cv::matMulDeriv(Mat A, Mat B, Mat& dABdA, Mat& dABdB)
    private static native void matMulDeriv_0(long A_nativeObj, long B_nativeObj, long dABdA_nativeObj, long dABdB_nativeObj);

    // C++:  void cv::projectPoints(vector_Point3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, vector_double distCoeffs, vector_Point2f& imagePoints, Mat& jacobian = Mat(), double aspectRatio = 0)
    private static native void projectPoints_0(long objectPoints_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long imagePoints_mat_nativeObj, long jacobian_nativeObj, double aspectRatio);
    private static native void projectPoints_1(long objectPoints_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long imagePoints_mat_nativeObj, long jacobian_nativeObj);
    private static native void projectPoints_2(long objectPoints_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long imagePoints_mat_nativeObj);

    // C++:  void cv::reprojectImageTo3D(Mat disparity, Mat& _3dImage, Mat Q, bool handleMissingValues = false, int ddepth = -1)
    private static native void reprojectImageTo3D_0(long disparity_nativeObj, long _3dImage_nativeObj, long Q_nativeObj, boolean handleMissingValues, int ddepth);
    private static native void reprojectImageTo3D_1(long disparity_nativeObj, long _3dImage_nativeObj, long Q_nativeObj, boolean handleMissingValues);
    private static native void reprojectImageTo3D_2(long disparity_nativeObj, long _3dImage_nativeObj, long Q_nativeObj);

    // C++:  void cv::solvePnPRefineLM(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat& rvec, Mat& tvec, TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, FLT_EPSILON))
    private static native void solvePnPRefineLM_0(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvec_nativeObj, long tvec_nativeObj, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native void solvePnPRefineLM_1(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvec_nativeObj, long tvec_nativeObj);

    // C++:  void cv::solvePnPRefineVVS(Mat objectPoints, Mat imagePoints, Mat cameraMatrix, Mat distCoeffs, Mat& rvec, Mat& tvec, TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, FLT_EPSILON), double VVSlambda = 1)
    private static native void solvePnPRefineVVS_0(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvec_nativeObj, long tvec_nativeObj, int criteria_type, int criteria_maxCount, double criteria_epsilon, double VVSlambda);
    private static native void solvePnPRefineVVS_1(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvec_nativeObj, long tvec_nativeObj, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native void solvePnPRefineVVS_2(long objectPoints_nativeObj, long imagePoints_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvec_nativeObj, long tvec_nativeObj);

    // C++:  void cv::stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, int flags = CALIB_ZERO_DISPARITY, double alpha = -1, Size newImageSize = Size(), Rect* validPixROI1 = 0, Rect* validPixROI2 = 0)
    private static native void stereoRectify_0(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double alpha, double newImageSize_width, double newImageSize_height, double[] validPixROI1_out, double[] validPixROI2_out);
    private static native void stereoRectify_1(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double alpha, double newImageSize_width, double newImageSize_height, double[] validPixROI1_out);
    private static native void stereoRectify_2(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double alpha, double newImageSize_width, double newImageSize_height);
    private static native void stereoRectify_3(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double alpha);
    private static native void stereoRectify_4(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags);
    private static native void stereoRectify_5(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj);

    // C++:  void cv::triangulatePoints(Mat projMatr1, Mat projMatr2, Mat projPoints1, Mat projPoints2, Mat& points4D)
    private static native void triangulatePoints_0(long projMatr1_nativeObj, long projMatr2_nativeObj, long projPoints1_nativeObj, long projPoints2_nativeObj, long points4D_nativeObj);

    // C++:  void cv::undistort(Mat src, Mat& dst, Mat cameraMatrix, Mat distCoeffs, Mat newCameraMatrix = Mat())
    private static native void undistort_0(long src_nativeObj, long dst_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long newCameraMatrix_nativeObj);
    private static native void undistort_1(long src_nativeObj, long dst_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj);

    // C++:  void cv::undistortPoints(Mat src, Mat& dst, Mat cameraMatrix, Mat distCoeffs, Mat R, Mat P, TermCriteria criteria)
    private static native void undistortPointsIter_0(long src_nativeObj, long dst_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long R_nativeObj, long P_nativeObj, int criteria_type, int criteria_maxCount, double criteria_epsilon);

    // C++:  void cv::undistortPoints(vector_Point2f src, vector_Point2f& dst, Mat cameraMatrix, Mat distCoeffs, Mat R = Mat(), Mat P = Mat())
    private static native void undistortPoints_0(long src_mat_nativeObj, long dst_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long R_nativeObj, long P_nativeObj);
    private static native void undistortPoints_1(long src_mat_nativeObj, long dst_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long R_nativeObj);
    private static native void undistortPoints_2(long src_mat_nativeObj, long dst_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_nativeObj);

    // C++:  void cv::validateDisparity(Mat& disparity, Mat cost, int minDisparity, int numberOfDisparities, int disp12MaxDisp = 1)
    private static native void validateDisparity_0(long disparity_nativeObj, long cost_nativeObj, int minDisparity, int numberOfDisparities, int disp12MaxDisp);
    private static native void validateDisparity_1(long disparity_nativeObj, long cost_nativeObj, int minDisparity, int numberOfDisparities);

    // C++:  void cv::fisheye::distortPoints(Mat undistorted, Mat& distorted, Mat K, Mat D, double alpha = 0)
    private static native void fisheye_distortPoints_0(long undistorted_nativeObj, long distorted_nativeObj, long K_nativeObj, long D_nativeObj, double alpha);
    private static native void fisheye_distortPoints_1(long undistorted_nativeObj, long distorted_nativeObj, long K_nativeObj, long D_nativeObj);

    // C++:  void cv::fisheye::estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat& P, double balance = 0.0, Size new_size = Size(), double fov_scale = 1.0)
    private static native void fisheye_estimateNewCameraMatrixForUndistortRectify_0(long K_nativeObj, long D_nativeObj, double image_size_width, double image_size_height, long R_nativeObj, long P_nativeObj, double balance, double new_size_width, double new_size_height, double fov_scale);
    private static native void fisheye_estimateNewCameraMatrixForUndistortRectify_1(long K_nativeObj, long D_nativeObj, double image_size_width, double image_size_height, long R_nativeObj, long P_nativeObj, double balance, double new_size_width, double new_size_height);
    private static native void fisheye_estimateNewCameraMatrixForUndistortRectify_2(long K_nativeObj, long D_nativeObj, double image_size_width, double image_size_height, long R_nativeObj, long P_nativeObj, double balance);
    private static native void fisheye_estimateNewCameraMatrixForUndistortRectify_3(long K_nativeObj, long D_nativeObj, double image_size_width, double image_size_height, long R_nativeObj, long P_nativeObj);

    // C++:  void cv::fisheye::initUndistortRectifyMap(Mat K, Mat D, Mat R, Mat P, Size size, int m1type, Mat& map1, Mat& map2)
    private static native void fisheye_initUndistortRectifyMap_0(long K_nativeObj, long D_nativeObj, long R_nativeObj, long P_nativeObj, double size_width, double size_height, int m1type, long map1_nativeObj, long map2_nativeObj);

    // C++:  void cv::fisheye::projectPoints(Mat objectPoints, Mat& imagePoints, Mat rvec, Mat tvec, Mat K, Mat D, double alpha = 0, Mat& jacobian = Mat())
    private static native void fisheye_projectPoints_0(long objectPoints_nativeObj, long imagePoints_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long K_nativeObj, long D_nativeObj, double alpha, long jacobian_nativeObj);
    private static native void fisheye_projectPoints_1(long objectPoints_nativeObj, long imagePoints_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long K_nativeObj, long D_nativeObj, double alpha);
    private static native void fisheye_projectPoints_2(long objectPoints_nativeObj, long imagePoints_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long K_nativeObj, long D_nativeObj);

    // C++:  void cv::fisheye::stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, int flags, Size newImageSize = Size(), double balance = 0.0, double fov_scale = 1.0)
    private static native void fisheye_stereoRectify_0(long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long tvec_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double newImageSize_width, double newImageSize_height, double balance, double fov_scale);
    private static native void fisheye_stereoRectify_1(long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long tvec_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double newImageSize_width, double newImageSize_height, double balance);
    private static native void fisheye_stereoRectify_2(long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long tvec_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double newImageSize_width, double newImageSize_height);
    private static native void fisheye_stereoRectify_3(long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long tvec_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags);

    // C++:  void cv::fisheye::undistortImage(Mat distorted, Mat& undistorted, Mat K, Mat D, Mat Knew = cv::Mat(), Size new_size = Size())
    private static native void fisheye_undistortImage_0(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj, long Knew_nativeObj, double new_size_width, double new_size_height);
    private static native void fisheye_undistortImage_1(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj, long Knew_nativeObj);
    private static native void fisheye_undistortImage_2(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj);

    // C++:  void cv::fisheye::undistortPoints(Mat distorted, Mat& undistorted, Mat K, Mat D, Mat R = Mat(), Mat P = Mat())
    private static native void fisheye_undistortPoints_0(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj, long R_nativeObj, long P_nativeObj);
    private static native void fisheye_undistortPoints_1(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj, long R_nativeObj);
    private static native void fisheye_undistortPoints_2(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj);

}
