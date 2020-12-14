//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.core;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.utils.Converters;

// C++: class Core

public class Core {
    // these constants are wrapped inside functions to prevent inlining
    private static String getVersion() { return "4.3.0"; }
    private static String getNativeLibraryName() { return "opencv_java430"; }
    private static int getVersionMajorJ() { return 4; }
    private static int getVersionMinorJ() { return 3; }
    private static int getVersionRevisionJ() { return 0; }
    private static String getVersionStatusJ() { return ""; }

    public static final String VERSION = getVersion();
    public static final String NATIVE_LIBRARY_NAME = getNativeLibraryName();
    public static final int VERSION_MAJOR = getVersionMajorJ();
    public static final int VERSION_MINOR = getVersionMinorJ();
    public static final int VERSION_REVISION = getVersionRevisionJ();
    public static final String VERSION_STATUS = getVersionStatusJ();

    private static final int
            CV_8U = 0,
            CV_8S = 1,
            CV_16U = 2,
            CV_16S = 3,
            CV_32S = 4,
            CV_32F = 5,
            CV_64F = 6,
            CV_USRTYPE1 = 7;


    // C++: enum DecompTypes
    public static final int
            DECOMP_LU = 0,
            DECOMP_SVD = 1,
            DECOMP_EIG = 2,
            DECOMP_CHOLESKY = 3,
            DECOMP_QR = 4,
            DECOMP_NORMAL = 16;


    // C++: enum BorderTypes
    public static final int
            BORDER_CONSTANT = 0,
            BORDER_REPLICATE = 1,
            BORDER_REFLECT = 2,
            BORDER_WRAP = 3,
            BORDER_REFLECT_101 = 4,
            BORDER_TRANSPARENT = 5,
            BORDER_REFLECT101 = BORDER_REFLECT_101,
            BORDER_DEFAULT = BORDER_REFLECT_101,
            BORDER_ISOLATED = 16;


    // C++: enum GemmFlags
    public static final int
            GEMM_1_T = 1,
            GEMM_2_T = 2,
            GEMM_3_T = 4;


    // C++: enum KmeansFlags
    public static final int
            KMEANS_RANDOM_CENTERS = 0,
            KMEANS_PP_CENTERS = 2,
            KMEANS_USE_INITIAL_LABELS = 1;


    // C++: enum CmpTypes
    public static final int
            CMP_EQ = 0,
            CMP_GT = 1,
            CMP_GE = 2,
            CMP_LT = 3,
            CMP_LE = 4,
            CMP_NE = 5;


    // C++: enum Flags
    public static final int
            PCA_DATA_AS_ROW = 0,
            PCA_DATA_AS_COL = 1,
            PCA_USE_AVG = 2;


    // C++: enum DftFlags
    public static final int
            DFT_INVERSE = 1,
            DFT_SCALE = 2,
            DFT_ROWS = 4,
            DFT_COMPLEX_OUTPUT = 16,
            DFT_REAL_OUTPUT = 32,
            DFT_COMPLEX_INPUT = 64,
            DCT_INVERSE = DFT_INVERSE,
            DCT_ROWS = DFT_ROWS;


    // C++: enum <unnamed>
    public static final int
            SVD_MODIFY_A = 1,
            SVD_NO_UV = 2,
            SVD_FULL_UV = 4,
            FILLED = -1,
            REDUCE_SUM = 0,
            REDUCE_AVG = 1,
            REDUCE_MAX = 2,
            REDUCE_MIN = 3,
            RNG_UNIFORM = 0,
            RNG_NORMAL = 1;


    // C++: enum CovarFlags
    public static final int
            COVAR_SCRAMBLED = 0,
            COVAR_NORMAL = 1,
            COVAR_USE_AVG = 2,
            COVAR_SCALE = 4,
            COVAR_ROWS = 8,
            COVAR_COLS = 16;


    // C++: enum SortFlags
    public static final int
            SORT_EVERY_ROW = 0,
            SORT_EVERY_COLUMN = 1,
            SORT_ASCENDING = 0,
            SORT_DESCENDING = 16;


    // C++: enum FormatType
    public static final int
            Formatter_FMT_DEFAULT = 0,
            Formatter_FMT_MATLAB = 1,
            Formatter_FMT_CSV = 2,
            Formatter_FMT_PYTHON = 3,
            Formatter_FMT_NUMPY = 4,
            Formatter_FMT_C = 5;


    // C++: enum Param
    public static final int
            Param_INT = 0,
            Param_BOOLEAN = 1,
            Param_REAL = 2,
            Param_STRING = 3,
            Param_MAT = 4,
            Param_MAT_VECTOR = 5,
            Param_ALGORITHM = 6,
            Param_FLOAT = 7,
            Param_UNSIGNED_INT = 8,
            Param_UINT64 = 9,
            Param_UCHAR = 11,
            Param_SCALAR = 12;


    // C++: enum NormTypes
    public static final int
            NORM_INF = 1,
            NORM_L1 = 2,
            NORM_L2 = 4,
            NORM_L2SQR = 5,
            NORM_HAMMING = 6,
            NORM_HAMMING2 = 7,
            NORM_TYPE_MASK = 7,
            NORM_RELATIVE = 8,
            NORM_MINMAX = 32;


    // C++: enum RotateFlags
    public static final int
            ROTATE_90_CLOCKWISE = 0,
            ROTATE_180 = 1,
            ROTATE_90_COUNTERCLOCKWISE = 2;


    // C++: enum Code
    public static final int
            StsOk = 0,
            StsBackTrace = -1,
            StsError = -2,
            StsInternal = -3,
            StsNoMem = -4,
            StsBadArg = -5,
            StsBadFunc = -6,
            StsNoConv = -7,
            StsAutoTrace = -8,
            HeaderIsNull = -9,
            BadImageSize = -10,
            BadOffset = -11,
            BadDataPtr = -12,
            BadStep = -13,
            BadModelOrChSeq = -14,
            BadNumChannels = -15,
            BadNumChannel1U = -16,
            BadDepth = -17,
            BadAlphaChannel = -18,
            BadOrder = -19,
            BadOrigin = -20,
            BadAlign = -21,
            BadCallBack = -22,
            BadTileSize = -23,
            BadCOI = -24,
            BadROISize = -25,
            MaskIsTiled = -26,
            StsNullPtr = -27,
            StsVecLengthErr = -28,
            StsFilterStructContentErr = -29,
            StsKernelStructContentErr = -30,
            StsFilterOffsetErr = -31,
            StsBadSize = -201,
            StsDivByZero = -202,
            StsInplaceNotSupported = -203,
            StsObjectNotFound = -204,
            StsUnmatchedFormats = -205,
            StsBadFlag = -206,
            StsBadPoint = -207,
            StsBadMask = -208,
            StsUnmatchedSizes = -209,
            StsUnsupportedFormat = -210,
            StsOutOfRange = -211,
            StsParseError = -212,
            StsNotImplemented = -213,
            StsBadMemBlock = -214,
            StsAssert = -215,
            GpuNotSupported = -216,
            GpuApiCallError = -217,
            OpenGlNotSupported = -218,
            OpenGlApiCallError = -219,
            OpenCLApiCallError = -220,
            OpenCLDoubleNotSupported = -221,
            OpenCLInitError = -222,
            OpenCLNoAMDBlasFft = -223;


    //
    // C++:  Scalar cv::mean(Mat src, Mat mask = Mat())
    //

    /**
     * Calculates an average (mean) of array elements.
     *
     * The function cv::mean calculates the mean value M of array elements,
     * independently for each channel, and return it:
     * \(\begin{array}{l} N =  \sum _{I: \; \texttt{mask} (I) \ne 0} 1 \\ M_c =  \left ( \sum _{I: \; \texttt{mask} (I) \ne 0}{ \texttt{mtx} (I)_c} \right )/N \end{array}\)
     * When all the mask elements are 0's, the function returns Scalar::all(0)
     * @param src input array that should have from 1 to 4 channels so that the result can be stored in
     * Scalar_ .
     * @param mask optional operation mask.
     * SEE:  countNonZero, meanStdDev, norm, minMaxLoc
     * @return automatically generated
     */
    public static Scalar mean(Mat src, Mat mask) {
        return new Scalar(mean_0(src.nativeObj, mask.nativeObj));
    }

    /**
     * Calculates an average (mean) of array elements.
     *
     * The function cv::mean calculates the mean value M of array elements,
     * independently for each channel, and return it:
     * \(\begin{array}{l} N =  \sum _{I: \; \texttt{mask} (I) \ne 0} 1 \\ M_c =  \left ( \sum _{I: \; \texttt{mask} (I) \ne 0}{ \texttt{mtx} (I)_c} \right )/N \end{array}\)
     * When all the mask elements are 0's, the function returns Scalar::all(0)
     * @param src input array that should have from 1 to 4 channels so that the result can be stored in
     * Scalar_ .
     * SEE:  countNonZero, meanStdDev, norm, minMaxLoc
     * @return automatically generated
     */
    public static Scalar mean(Mat src) {
        return new Scalar(mean_1(src.nativeObj));
    }


    //
    // C++:  Scalar cv::sum(Mat src)
    //

    /**
     * Calculates the sum of array elements.
     *
     * The function cv::sum calculates and returns the sum of array elements,
     * independently for each channel.
     * @param src input array that must have from 1 to 4 channels.
     * SEE:  countNonZero, mean, meanStdDev, norm, minMaxLoc, reduce
     * @return automatically generated
     */
    public static Scalar sumElems(Mat src) {
        return new Scalar(sumElems_0(src.nativeObj));
    }


    //
    // C++:  Scalar cv::trace(Mat mtx)
    //

    /**
     * Returns the trace of a matrix.
     *
     * The function cv::trace returns the sum of the diagonal elements of the
     * matrix mtx .
     * \(\mathrm{tr} ( \texttt{mtx} ) =  \sum _i  \texttt{mtx} (i,i)\)
     * @param mtx input matrix.
     * @return automatically generated
     */
    public static Scalar trace(Mat mtx) {
        return new Scalar(trace_0(mtx.nativeObj));
    }


    //
    // C++:  String cv::getBuildInformation()
    //

    /**
     * Returns full configuration time cmake output.
     *
     * Returned value is raw cmake output including version control system revision, compiler version,
     * compiler flags, enabled modules and third party libraries, etc. Output format depends on target
     * architecture.
     * @return automatically generated
     */
    public static String getBuildInformation() {
        return getBuildInformation_0();
    }


    //
    // C++:  String cv::getHardwareFeatureName(int feature)
    //

    /**
     * Returns feature name by ID
     *
     * Returns empty string if feature is not defined
     * @param feature automatically generated
     * @return automatically generated
     */
    public static String getHardwareFeatureName(int feature) {
        return getHardwareFeatureName_0(feature);
    }


    //
    // C++:  String cv::getVersionString()
    //

    /**
     * Returns library version string
     *
     * For example "3.4.1-dev".
     *
     * SEE: getMajorVersion, getMinorVersion, getRevisionVersion
     * @return automatically generated
     */
    public static String getVersionString() {
        return getVersionString_0();
    }


    //
    // C++:  String cv::ipp::getIppVersion()
    //

    public static String getIppVersion() {
        return getIppVersion_0();
    }


    //
    // C++:  String cv::samples::findFile(String relative_path, bool required = true, bool silentMode = false)
    //

    /**
     * Try to find requested data file
     *
     * Search directories:
     *
     * 1. Directories passed via {@code addSamplesDataSearchPath()}
     * 2. OPENCV_SAMPLES_DATA_PATH_HINT environment variable
     * 3. OPENCV_SAMPLES_DATA_PATH environment variable
     *    If parameter value is not empty and nothing is found then stop searching.
     * 4. Detects build/install path based on:
     *    a. current working directory (CWD)
     *    b. and/or binary module location (opencv_core/opencv_world, doesn't work with static linkage)
     * 5. Scan {@code &lt;source&gt;/{,data,samples/data}} directories if build directory is detected or the current directory is in source tree.
     * 6. Scan {@code &lt;install&gt;/share/OpenCV} directory if install directory is detected.
     *
     * SEE: cv::utils::findDataFile
     *
     * @param relative_path Relative path to data file
     * @param required Specify "file not found" handling.
     *        If true, function prints information message and raises cv::Exception.
     *        If false, function returns empty result
     * @param silentMode Disables messages
     * @return Returns path (absolute or relative to the current directory) or empty string if file is not found
     */
    public static String findFile(String relative_path, boolean required, boolean silentMode) {
        return findFile_0(relative_path, required, silentMode);
    }

    /**
     * Try to find requested data file
     *
     * Search directories:
     *
     * 1. Directories passed via {@code addSamplesDataSearchPath()}
     * 2. OPENCV_SAMPLES_DATA_PATH_HINT environment variable
     * 3. OPENCV_SAMPLES_DATA_PATH environment variable
     *    If parameter value is not empty and nothing is found then stop searching.
     * 4. Detects build/install path based on:
     *    a. current working directory (CWD)
     *    b. and/or binary module location (opencv_core/opencv_world, doesn't work with static linkage)
     * 5. Scan {@code &lt;source&gt;/{,data,samples/data}} directories if build directory is detected or the current directory is in source tree.
     * 6. Scan {@code &lt;install&gt;/share/OpenCV} directory if install directory is detected.
     *
     * SEE: cv::utils::findDataFile
     *
     * @param relative_path Relative path to data file
     * @param required Specify "file not found" handling.
     *        If true, function prints information message and raises cv::Exception.
     *        If false, function returns empty result
     * @return Returns path (absolute or relative to the current directory) or empty string if file is not found
     */
    public static String findFile(String relative_path, boolean required) {
        return findFile_1(relative_path, required);
    }

    /**
     * Try to find requested data file
     *
     * Search directories:
     *
     * 1. Directories passed via {@code addSamplesDataSearchPath()}
     * 2. OPENCV_SAMPLES_DATA_PATH_HINT environment variable
     * 3. OPENCV_SAMPLES_DATA_PATH environment variable
     *    If parameter value is not empty and nothing is found then stop searching.
     * 4. Detects build/install path based on:
     *    a. current working directory (CWD)
     *    b. and/or binary module location (opencv_core/opencv_world, doesn't work with static linkage)
     * 5. Scan {@code &lt;source&gt;/{,data,samples/data}} directories if build directory is detected or the current directory is in source tree.
     * 6. Scan {@code &lt;install&gt;/share/OpenCV} directory if install directory is detected.
     *
     * SEE: cv::utils::findDataFile
     *
     * @param relative_path Relative path to data file
     *        If true, function prints information message and raises cv::Exception.
     *        If false, function returns empty result
     * @return Returns path (absolute or relative to the current directory) or empty string if file is not found
     */
    public static String findFile(String relative_path) {
        return findFile_2(relative_path);
    }


    //
    // C++:  String cv::samples::findFileOrKeep(String relative_path, bool silentMode = false)
    //

    public static String findFileOrKeep(String relative_path, boolean silentMode) {
        return findFileOrKeep_0(relative_path, silentMode);
    }

    public static String findFileOrKeep(String relative_path) {
        return findFileOrKeep_1(relative_path);
    }


    //
    // C++:  bool cv::checkRange(Mat a, bool quiet = true,  _hidden_ * pos = 0, double minVal = -DBL_MAX, double maxVal = DBL_MAX)
    //

    /**
     * Checks every element of an input array for invalid values.
     *
     * The function cv::checkRange checks that every array element is neither NaN nor infinite. When minVal &gt;
     * <ul>
     *   <li>
     * DBL_MAX and maxVal &lt; DBL_MAX, the function also checks that each value is between minVal and
     * maxVal. In case of multi-channel arrays, each channel is processed independently. If some values
     * are out of range, position of the first outlier is stored in pos (when pos != NULL). Then, the
     * function either returns false (when quiet=true) or throws an exception.
     * @param a input array.
     * @param quiet a flag, indicating whether the functions quietly return false when the array elements
     * are out of range or they throw an exception.
     * elements.
     * @param minVal inclusive lower boundary of valid values range.
     * @param maxVal exclusive upper boundary of valid values range.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean checkRange(Mat a, boolean quiet, double minVal, double maxVal) {
        return checkRange_0(a.nativeObj, quiet, minVal, maxVal);
    }

    /**
     * Checks every element of an input array for invalid values.
     *
     * The function cv::checkRange checks that every array element is neither NaN nor infinite. When minVal &gt;
     * <ul>
     *   <li>
     * DBL_MAX and maxVal &lt; DBL_MAX, the function also checks that each value is between minVal and
     * maxVal. In case of multi-channel arrays, each channel is processed independently. If some values
     * are out of range, position of the first outlier is stored in pos (when pos != NULL). Then, the
     * function either returns false (when quiet=true) or throws an exception.
     * @param a input array.
     * @param quiet a flag, indicating whether the functions quietly return false when the array elements
     * are out of range or they throw an exception.
     * elements.
     * @param minVal inclusive lower boundary of valid values range.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean checkRange(Mat a, boolean quiet, double minVal) {
        return checkRange_1(a.nativeObj, quiet, minVal);
    }

    /**
     * Checks every element of an input array for invalid values.
     *
     * The function cv::checkRange checks that every array element is neither NaN nor infinite. When minVal &gt;
     * <ul>
     *   <li>
     * DBL_MAX and maxVal &lt; DBL_MAX, the function also checks that each value is between minVal and
     * maxVal. In case of multi-channel arrays, each channel is processed independently. If some values
     * are out of range, position of the first outlier is stored in pos (when pos != NULL). Then, the
     * function either returns false (when quiet=true) or throws an exception.
     * @param a input array.
     * @param quiet a flag, indicating whether the functions quietly return false when the array elements
     * are out of range or they throw an exception.
     * elements.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean checkRange(Mat a, boolean quiet) {
        return checkRange_2(a.nativeObj, quiet);
    }

    /**
     * Checks every element of an input array for invalid values.
     *
     * The function cv::checkRange checks that every array element is neither NaN nor infinite. When minVal &gt;
     * <ul>
     *   <li>
     * DBL_MAX and maxVal &lt; DBL_MAX, the function also checks that each value is between minVal and
     * maxVal. In case of multi-channel arrays, each channel is processed independently. If some values
     * are out of range, position of the first outlier is stored in pos (when pos != NULL). Then, the
     * function either returns false (when quiet=true) or throws an exception.
     * @param a input array.
     * are out of range or they throw an exception.
     * elements.
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static boolean checkRange(Mat a) {
        return checkRange_4(a.nativeObj);
    }


    //
    // C++:  bool cv::eigen(Mat src, Mat& eigenvalues, Mat& eigenvectors = Mat())
    //

    /**
     * Calculates eigenvalues and eigenvectors of a symmetric matrix.
     *
     * The function cv::eigen calculates just eigenvalues, or eigenvalues and eigenvectors of the symmetric
     * matrix src:
     * <code>
     *     src*eigenvectors.row(i).t() = eigenvalues.at&lt;srcType&gt;(i)*eigenvectors.row(i).t()
     * </code>
     *
     * <b>Note:</b> Use cv::eigenNonSymmetric for calculation of real eigenvalues and eigenvectors of non-symmetric matrix.
     *
     * @param src input matrix that must have CV_32FC1 or CV_64FC1 type, square size and be symmetrical
     * (src ^T^ == src).
     * @param eigenvalues output vector of eigenvalues of the same type as src; the eigenvalues are stored
     * in the descending order.
     * @param eigenvectors output matrix of eigenvectors; it has the same size and type as src; the
     * eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding
     * eigenvalues.
     * SEE: eigenNonSymmetric, completeSymm , PCA
     * @return automatically generated
     */
    public static boolean eigen(Mat src, Mat eigenvalues, Mat eigenvectors) {
        return eigen_0(src.nativeObj, eigenvalues.nativeObj, eigenvectors.nativeObj);
    }

    /**
     * Calculates eigenvalues and eigenvectors of a symmetric matrix.
     *
     * The function cv::eigen calculates just eigenvalues, or eigenvalues and eigenvectors of the symmetric
     * matrix src:
     * <code>
     *     src*eigenvectors.row(i).t() = eigenvalues.at&lt;srcType&gt;(i)*eigenvectors.row(i).t()
     * </code>
     *
     * <b>Note:</b> Use cv::eigenNonSymmetric for calculation of real eigenvalues and eigenvectors of non-symmetric matrix.
     *
     * @param src input matrix that must have CV_32FC1 or CV_64FC1 type, square size and be symmetrical
     * (src ^T^ == src).
     * @param eigenvalues output vector of eigenvalues of the same type as src; the eigenvalues are stored
     * in the descending order.
     * eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding
     * eigenvalues.
     * SEE: eigenNonSymmetric, completeSymm , PCA
     * @return automatically generated
     */
    public static boolean eigen(Mat src, Mat eigenvalues) {
        return eigen_1(src.nativeObj, eigenvalues.nativeObj);
    }


    //
    // C++:  bool cv::solve(Mat src1, Mat src2, Mat& dst, int flags = DECOMP_LU)
    //

    /**
     * Solves one or more linear systems or least-squares problems.
     *
     * The function cv::solve solves a linear system or least-squares problem (the
     * latter is possible with SVD or QR methods, or by specifying the flag
     * #DECOMP_NORMAL ):
     * \(\texttt{dst} =  \arg \min _X \| \texttt{src1} \cdot \texttt{X} -  \texttt{src2} \|\)
     *
     * If #DECOMP_LU or #DECOMP_CHOLESKY method is used, the function returns 1
     * if src1 (or \(\texttt{src1}^T\texttt{src1}\) ) is non-singular. Otherwise,
     * it returns 0. In the latter case, dst is not valid. Other methods find a
     * pseudo-solution in case of a singular left-hand side part.
     *
     * <b>Note:</b> If you want to find a unity-norm solution of an under-defined
     * singular system \(\texttt{src1}\cdot\texttt{dst}=0\) , the function solve
     * will not do the work. Use SVD::solveZ instead.
     *
     * @param src1 input matrix on the left-hand side of the system.
     * @param src2 input matrix on the right-hand side of the system.
     * @param dst output solution.
     * @param flags solution (matrix inversion) method (#DecompTypes)
     * SEE: invert, SVD, eigen
     * @return automatically generated
     */
    public static boolean solve(Mat src1, Mat src2, Mat dst, int flags) {
        return solve_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, flags);
    }

    /**
     * Solves one or more linear systems or least-squares problems.
     *
     * The function cv::solve solves a linear system or least-squares problem (the
     * latter is possible with SVD or QR methods, or by specifying the flag
     * #DECOMP_NORMAL ):
     * \(\texttt{dst} =  \arg \min _X \| \texttt{src1} \cdot \texttt{X} -  \texttt{src2} \|\)
     *
     * If #DECOMP_LU or #DECOMP_CHOLESKY method is used, the function returns 1
     * if src1 (or \(\texttt{src1}^T\texttt{src1}\) ) is non-singular. Otherwise,
     * it returns 0. In the latter case, dst is not valid. Other methods find a
     * pseudo-solution in case of a singular left-hand side part.
     *
     * <b>Note:</b> If you want to find a unity-norm solution of an under-defined
     * singular system \(\texttt{src1}\cdot\texttt{dst}=0\) , the function solve
     * will not do the work. Use SVD::solveZ instead.
     *
     * @param src1 input matrix on the left-hand side of the system.
     * @param src2 input matrix on the right-hand side of the system.
     * @param dst output solution.
     * SEE: invert, SVD, eigen
     * @return automatically generated
     */
    public static boolean solve(Mat src1, Mat src2, Mat dst) {
        return solve_1(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  bool cv::ipp::useIPP()
    //

    /**
     * proxy for hal::Cholesky
     * @return automatically generated
     */
    public static boolean useIPP() {
        return useIPP_0();
    }


    //
    // C++:  bool cv::ipp::useIPP_NotExact()
    //

    public static boolean useIPP_NotExact() {
        return useIPP_NotExact_0();
    }


    //
    // C++:  double cv::Mahalanobis(Mat v1, Mat v2, Mat icovar)
    //

    /**
     * Calculates the Mahalanobis distance between two vectors.
     *
     * The function cv::Mahalanobis calculates and returns the weighted distance between two vectors:
     * \(d( \texttt{vec1} , \texttt{vec2} )= \sqrt{\sum_{i,j}{\texttt{icovar(i,j)}\cdot(\texttt{vec1}(I)-\texttt{vec2}(I))\cdot(\texttt{vec1(j)}-\texttt{vec2(j)})} }\)
     * The covariance matrix may be calculated using the #calcCovarMatrix function and then inverted using
     * the invert function (preferably using the #DECOMP_SVD method, as the most accurate).
     * @param v1 first 1D input vector.
     * @param v2 second 1D input vector.
     * @param icovar inverse covariance matrix.
     * @return automatically generated
     */
    public static double Mahalanobis(Mat v1, Mat v2, Mat icovar) {
        return Mahalanobis_0(v1.nativeObj, v2.nativeObj, icovar.nativeObj);
    }


    //
    // C++:  double cv::PSNR(Mat src1, Mat src2, double R = 255.)
    //

    /**
     * Computes the Peak Signal-to-Noise Ratio (PSNR) image quality metric.
     *
     * This function calculates the Peak Signal-to-Noise Ratio (PSNR) image quality metric in decibels (dB),
     * between two input arrays src1 and src2. The arrays must have the same type.
     *
     * The PSNR is calculated as follows:
     *
     * \(
     * \texttt{PSNR} = 10 \cdot \log_{10}{\left( \frac{R^2}{MSE} \right) }
     * \)
     *
     * where R is the maximum integer value of depth (e.g. 255 in the case of CV_8U data)
     * and MSE is the mean squared error between the two arrays.
     *
     * @param src1 first input array.
     * @param src2 second input array of the same size as src1.
     * @param R the maximum pixel value (255 by default)
     * @return automatically generated
     */
    public static double PSNR(Mat src1, Mat src2, double R) {
        return PSNR_0(src1.nativeObj, src2.nativeObj, R);
    }

    /**
     * Computes the Peak Signal-to-Noise Ratio (PSNR) image quality metric.
     *
     * This function calculates the Peak Signal-to-Noise Ratio (PSNR) image quality metric in decibels (dB),
     * between two input arrays src1 and src2. The arrays must have the same type.
     *
     * The PSNR is calculated as follows:
     *
     * \(
     * \texttt{PSNR} = 10 \cdot \log_{10}{\left( \frac{R^2}{MSE} \right) }
     * \)
     *
     * where R is the maximum integer value of depth (e.g. 255 in the case of CV_8U data)
     * and MSE is the mean squared error between the two arrays.
     *
     * @param src1 first input array.
     * @param src2 second input array of the same size as src1.
     * @return automatically generated
     */
    public static double PSNR(Mat src1, Mat src2) {
        return PSNR_1(src1.nativeObj, src2.nativeObj);
    }


    //
    // C++:  double cv::determinant(Mat mtx)
    //

    /**
     * Returns the determinant of a square floating-point matrix.
     *
     * The function cv::determinant calculates and returns the determinant of the
     * specified matrix. For small matrices ( mtx.cols=mtx.rows&lt;=3 ), the
     * direct method is used. For larger matrices, the function uses LU
     * factorization with partial pivoting.
     *
     * For symmetric positively-determined matrices, it is also possible to use
     * eigen decomposition to calculate the determinant.
     * @param mtx input matrix that must have CV_32FC1 or CV_64FC1 type and
     * square size.
     * SEE: trace, invert, solve, eigen, REF: MatrixExpressions
     * @return automatically generated
     */
    public static double determinant(Mat mtx) {
        return determinant_0(mtx.nativeObj);
    }


    //
    // C++:  double cv::getTickFrequency()
    //

    /**
     * Returns the number of ticks per second.
     *
     * The function returns the number of ticks per second. That is, the following code computes the
     * execution time in seconds:
     * <code>
     *     double t = (double)getTickCount();
     *     // do something ...
     *     t = ((double)getTickCount() - t)/getTickFrequency();
     * </code>
     * SEE: getTickCount, TickMeter
     * @return automatically generated
     */
    public static double getTickFrequency() {
        return getTickFrequency_0();
    }


    //
    // C++:  double cv::invert(Mat src, Mat& dst, int flags = DECOMP_LU)
    //

    /**
     * Finds the inverse or pseudo-inverse of a matrix.
     *
     * The function cv::invert inverts the matrix src and stores the result in dst
     * . When the matrix src is singular or non-square, the function calculates
     * the pseudo-inverse matrix (the dst matrix) so that norm(src\*dst - I) is
     * minimal, where I is an identity matrix.
     *
     * In case of the #DECOMP_LU method, the function returns non-zero value if
     * the inverse has been successfully calculated and 0 if src is singular.
     *
     * In case of the #DECOMP_SVD method, the function returns the inverse
     * condition number of src (the ratio of the smallest singular value to the
     * largest singular value) and 0 if src is singular. The SVD method
     * calculates a pseudo-inverse matrix if src is singular.
     *
     * Similarly to #DECOMP_LU, the method #DECOMP_CHOLESKY works only with
     * non-singular square matrices that should also be symmetrical and
     * positively defined. In this case, the function stores the inverted
     * matrix in dst and returns non-zero. Otherwise, it returns 0.
     *
     * @param src input floating-point M x N matrix.
     * @param dst output matrix of N x M size and the same type as src.
     * @param flags inversion method (cv::DecompTypes)
     * SEE: solve, SVD
     * @return automatically generated
     */
    public static double invert(Mat src, Mat dst, int flags) {
        return invert_0(src.nativeObj, dst.nativeObj, flags);
    }

    /**
     * Finds the inverse or pseudo-inverse of a matrix.
     *
     * The function cv::invert inverts the matrix src and stores the result in dst
     * . When the matrix src is singular or non-square, the function calculates
     * the pseudo-inverse matrix (the dst matrix) so that norm(src\*dst - I) is
     * minimal, where I is an identity matrix.
     *
     * In case of the #DECOMP_LU method, the function returns non-zero value if
     * the inverse has been successfully calculated and 0 if src is singular.
     *
     * In case of the #DECOMP_SVD method, the function returns the inverse
     * condition number of src (the ratio of the smallest singular value to the
     * largest singular value) and 0 if src is singular. The SVD method
     * calculates a pseudo-inverse matrix if src is singular.
     *
     * Similarly to #DECOMP_LU, the method #DECOMP_CHOLESKY works only with
     * non-singular square matrices that should also be symmetrical and
     * positively defined. In this case, the function stores the inverted
     * matrix in dst and returns non-zero. Otherwise, it returns 0.
     *
     * @param src input floating-point M x N matrix.
     * @param dst output matrix of N x M size and the same type as src.
     * SEE: solve, SVD
     * @return automatically generated
     */
    public static double invert(Mat src, Mat dst) {
        return invert_1(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  double cv::kmeans(Mat data, int K, Mat& bestLabels, TermCriteria criteria, int attempts, int flags, Mat& centers = Mat())
    //

    /**
     * Finds centers of clusters and groups input samples around the clusters.
     *
     * The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters
     * and groups the input samples around the clusters. As an output, \(\texttt{bestLabels}_i\) contains a
     * 0-based cluster index for the sample stored in the \(i^{th}\) row of the samples matrix.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *    (Python) An example on K-means clustering can be found at
     *     opencv_source_code/samples/python/kmeans.py
     * @param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
     * Examples of this array can be:
     *   </li>
     *   <li>
     *    Mat points(count, 2, CV_32F);
     *   </li>
     *   <li>
     *    Mat points(count, 1, CV_32FC2);
     *   </li>
     *   <li>
     *    Mat points(1, count, CV_32FC2);
     *   </li>
     *   <li>
     *    std::vector&lt;cv::Point2f&gt; points(sampleCount);
     * @param K Number of clusters to split the set by.
     * @param bestLabels Input/output integer array that stores the cluster indices for every sample.
     * @param criteria The algorithm termination criteria, that is, the maximum number of iterations and/or
     * the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster
     * centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
     * @param attempts Flag to specify the number of times the algorithm is executed using different
     * initial labellings. The algorithm returns the labels that yield the best compactness (see the last
     * function parameter).
     * @param flags Flag that can take values of cv::KmeansFlags
     * @param centers Output matrix of the cluster centers, one row per each cluster center.
     * @return The function returns the compactness measure that is computed as
     * \(\sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2\)
     * after every attempt. The best (minimum) value is chosen and the corresponding labels and the
     * compactness value are returned by the function. Basically, you can use only the core of the
     * function, set the number of attempts to 1, initialize labels each time using a custom algorithm,
     * pass them with the ( flags = #KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best
     * (most-compact) clustering.
     *   </li>
     * </ul>
     */
    public static double kmeans(Mat data, int K, Mat bestLabels, TermCriteria criteria, int attempts, int flags, Mat centers) {
        return kmeans_0(data.nativeObj, K, bestLabels.nativeObj, criteria.type, criteria.maxCount, criteria.epsilon, attempts, flags, centers.nativeObj);
    }

    /**
     * Finds centers of clusters and groups input samples around the clusters.
     *
     * The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters
     * and groups the input samples around the clusters. As an output, \(\texttt{bestLabels}_i\) contains a
     * 0-based cluster index for the sample stored in the \(i^{th}\) row of the samples matrix.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *    (Python) An example on K-means clustering can be found at
     *     opencv_source_code/samples/python/kmeans.py
     * @param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
     * Examples of this array can be:
     *   </li>
     *   <li>
     *    Mat points(count, 2, CV_32F);
     *   </li>
     *   <li>
     *    Mat points(count, 1, CV_32FC2);
     *   </li>
     *   <li>
     *    Mat points(1, count, CV_32FC2);
     *   </li>
     *   <li>
     *    std::vector&lt;cv::Point2f&gt; points(sampleCount);
     * @param K Number of clusters to split the set by.
     * @param bestLabels Input/output integer array that stores the cluster indices for every sample.
     * @param criteria The algorithm termination criteria, that is, the maximum number of iterations and/or
     * the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster
     * centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
     * @param attempts Flag to specify the number of times the algorithm is executed using different
     * initial labellings. The algorithm returns the labels that yield the best compactness (see the last
     * function parameter).
     * @param flags Flag that can take values of cv::KmeansFlags
     * @return The function returns the compactness measure that is computed as
     * \(\sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2\)
     * after every attempt. The best (minimum) value is chosen and the corresponding labels and the
     * compactness value are returned by the function. Basically, you can use only the core of the
     * function, set the number of attempts to 1, initialize labels each time using a custom algorithm,
     * pass them with the ( flags = #KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best
     * (most-compact) clustering.
     *   </li>
     * </ul>
     */
    public static double kmeans(Mat data, int K, Mat bestLabels, TermCriteria criteria, int attempts, int flags) {
        return kmeans_1(data.nativeObj, K, bestLabels.nativeObj, criteria.type, criteria.maxCount, criteria.epsilon, attempts, flags);
    }


    //
    // C++:  double cv::norm(Mat src1, Mat src2, int normType = NORM_L2, Mat mask = Mat())
    //

    /**
     * Calculates an absolute difference norm or a relative difference norm.
     *
     * This version of cv::norm calculates the absolute difference norm
     * or the relative difference norm of arrays src1 and src2.
     * The type of norm to calculate is specified using #NormTypes.
     *
     * @param src1 first input array.
     * @param src2 second input array of the same size and the same type as src1.
     * @param normType type of the norm (see #NormTypes).
     * @param mask optional operation mask; it must have the same size as src1 and CV_8UC1 type.
     * @return automatically generated
     */
    public static double norm(Mat src1, Mat src2, int normType, Mat mask) {
        return norm_0(src1.nativeObj, src2.nativeObj, normType, mask.nativeObj);
    }

    /**
     * Calculates an absolute difference norm or a relative difference norm.
     *
     * This version of cv::norm calculates the absolute difference norm
     * or the relative difference norm of arrays src1 and src2.
     * The type of norm to calculate is specified using #NormTypes.
     *
     * @param src1 first input array.
     * @param src2 second input array of the same size and the same type as src1.
     * @param normType type of the norm (see #NormTypes).
     * @return automatically generated
     */
    public static double norm(Mat src1, Mat src2, int normType) {
        return norm_1(src1.nativeObj, src2.nativeObj, normType);
    }

    /**
     * Calculates an absolute difference norm or a relative difference norm.
     *
     * This version of cv::norm calculates the absolute difference norm
     * or the relative difference norm of arrays src1 and src2.
     * The type of norm to calculate is specified using #NormTypes.
     *
     * @param src1 first input array.
     * @param src2 second input array of the same size and the same type as src1.
     * @return automatically generated
     */
    public static double norm(Mat src1, Mat src2) {
        return norm_2(src1.nativeObj, src2.nativeObj);
    }


    //
    // C++:  double cv::norm(Mat src1, int normType = NORM_L2, Mat mask = Mat())
    //

    /**
     * Calculates the  absolute norm of an array.
     *
     * This version of #norm calculates the absolute norm of src1. The type of norm to calculate is specified using #NormTypes.
     *
     * As example for one array consider the function \(r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]\).
     * The \( L_{1}, L_{2} \) and \( L_{\infty} \) norm for the sample value \(r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}\)
     * is calculated as follows
     * \(align*}
     *     \| r(-1) \|_{L_1} &amp;= |-1| + |2| = 3 \\
     *     \| r(-1) \|_{L_2} &amp;= \sqrt{(-1)^{2} + (2)^{2}} = \sqrt{5} \\
     *     \| r(-1) \|_{L_\infty} &amp;= \max(|-1|,|2|) = 2
     * \)
     * and for \(r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}\) the calculation is
     * \(align*}
     *     \| r(0.5) \|_{L_1} &amp;= |0.5| + |0.5| = 1 \\
     *     \| r(0.5) \|_{L_2} &amp;= \sqrt{(0.5)^{2} + (0.5)^{2}} = \sqrt{0.5} \\
     *     \| r(0.5) \|_{L_\infty} &amp;= \max(|0.5|,|0.5|) = 0.5.
     * \)
     * The following graphic shows all values for the three norm functions \(\| r(x) \|_{L_1}, \| r(x) \|_{L_2}\) and \(\| r(x) \|_{L_\infty}\).
     * It is notable that the \( L_{1} \) norm forms the upper and the \( L_{\infty} \) norm forms the lower border for the example function \( r(x) \).
     * ![Graphs for the different norm functions from the above example](pics/NormTypes_OneArray_1-2-INF.png)
     *
     * When the mask parameter is specified and it is not empty, the norm is
     *
     * If normType is not specified, #NORM_L2 is used.
     * calculated only over the region specified by the mask.
     *
     * Multi-channel input arrays are treated as single-channel arrays, that is,
     * the results for all channels are combined.
     *
     * Hamming norms can only be calculated with CV_8U depth arrays.
     *
     * @param src1 first input array.
     * @param normType type of the norm (see #NormTypes).
     * @param mask optional operation mask; it must have the same size as src1 and CV_8UC1 type.
     * @return automatically generated
     */
    public static double norm(Mat src1, int normType, Mat mask) {
        return norm_3(src1.nativeObj, normType, mask.nativeObj);
    }

    /**
     * Calculates the  absolute norm of an array.
     *
     * This version of #norm calculates the absolute norm of src1. The type of norm to calculate is specified using #NormTypes.
     *
     * As example for one array consider the function \(r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]\).
     * The \( L_{1}, L_{2} \) and \( L_{\infty} \) norm for the sample value \(r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}\)
     * is calculated as follows
     * \(align*}
     *     \| r(-1) \|_{L_1} &amp;= |-1| + |2| = 3 \\
     *     \| r(-1) \|_{L_2} &amp;= \sqrt{(-1)^{2} + (2)^{2}} = \sqrt{5} \\
     *     \| r(-1) \|_{L_\infty} &amp;= \max(|-1|,|2|) = 2
     * \)
     * and for \(r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}\) the calculation is
     * \(align*}
     *     \| r(0.5) \|_{L_1} &amp;= |0.5| + |0.5| = 1 \\
     *     \| r(0.5) \|_{L_2} &amp;= \sqrt{(0.5)^{2} + (0.5)^{2}} = \sqrt{0.5} \\
     *     \| r(0.5) \|_{L_\infty} &amp;= \max(|0.5|,|0.5|) = 0.5.
     * \)
     * The following graphic shows all values for the three norm functions \(\| r(x) \|_{L_1}, \| r(x) \|_{L_2}\) and \(\| r(x) \|_{L_\infty}\).
     * It is notable that the \( L_{1} \) norm forms the upper and the \( L_{\infty} \) norm forms the lower border for the example function \( r(x) \).
     * ![Graphs for the different norm functions from the above example](pics/NormTypes_OneArray_1-2-INF.png)
     *
     * When the mask parameter is specified and it is not empty, the norm is
     *
     * If normType is not specified, #NORM_L2 is used.
     * calculated only over the region specified by the mask.
     *
     * Multi-channel input arrays are treated as single-channel arrays, that is,
     * the results for all channels are combined.
     *
     * Hamming norms can only be calculated with CV_8U depth arrays.
     *
     * @param src1 first input array.
     * @param normType type of the norm (see #NormTypes).
     * @return automatically generated
     */
    public static double norm(Mat src1, int normType) {
        return norm_4(src1.nativeObj, normType);
    }

    /**
     * Calculates the  absolute norm of an array.
     *
     * This version of #norm calculates the absolute norm of src1. The type of norm to calculate is specified using #NormTypes.
     *
     * As example for one array consider the function \(r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]\).
     * The \( L_{1}, L_{2} \) and \( L_{\infty} \) norm for the sample value \(r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}\)
     * is calculated as follows
     * \(align*}
     *     \| r(-1) \|_{L_1} &amp;= |-1| + |2| = 3 \\
     *     \| r(-1) \|_{L_2} &amp;= \sqrt{(-1)^{2} + (2)^{2}} = \sqrt{5} \\
     *     \| r(-1) \|_{L_\infty} &amp;= \max(|-1|,|2|) = 2
     * \)
     * and for \(r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}\) the calculation is
     * \(align*}
     *     \| r(0.5) \|_{L_1} &amp;= |0.5| + |0.5| = 1 \\
     *     \| r(0.5) \|_{L_2} &amp;= \sqrt{(0.5)^{2} + (0.5)^{2}} = \sqrt{0.5} \\
     *     \| r(0.5) \|_{L_\infty} &amp;= \max(|0.5|,|0.5|) = 0.5.
     * \)
     * The following graphic shows all values for the three norm functions \(\| r(x) \|_{L_1}, \| r(x) \|_{L_2}\) and \(\| r(x) \|_{L_\infty}\).
     * It is notable that the \( L_{1} \) norm forms the upper and the \( L_{\infty} \) norm forms the lower border for the example function \( r(x) \).
     * ![Graphs for the different norm functions from the above example](pics/NormTypes_OneArray_1-2-INF.png)
     *
     * When the mask parameter is specified and it is not empty, the norm is
     *
     * If normType is not specified, #NORM_L2 is used.
     * calculated only over the region specified by the mask.
     *
     * Multi-channel input arrays are treated as single-channel arrays, that is,
     * the results for all channels are combined.
     *
     * Hamming norms can only be calculated with CV_8U depth arrays.
     *
     * @param src1 first input array.
     * @return automatically generated
     */
    public static double norm(Mat src1) {
        return norm_5(src1.nativeObj);
    }


    //
    // C++:  double cv::solvePoly(Mat coeffs, Mat& roots, int maxIters = 300)
    //

    /**
     * Finds the real or complex roots of a polynomial equation.
     *
     * The function cv::solvePoly finds real and complex roots of a polynomial equation:
     * \(\texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0\)
     * @param coeffs array of polynomial coefficients.
     * @param roots output (complex) array of roots.
     * @param maxIters maximum number of iterations the algorithm does.
     * @return automatically generated
     */
    public static double solvePoly(Mat coeffs, Mat roots, int maxIters) {
        return solvePoly_0(coeffs.nativeObj, roots.nativeObj, maxIters);
    }

    /**
     * Finds the real or complex roots of a polynomial equation.
     *
     * The function cv::solvePoly finds real and complex roots of a polynomial equation:
     * \(\texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0\)
     * @param coeffs array of polynomial coefficients.
     * @param roots output (complex) array of roots.
     * @return automatically generated
     */
    public static double solvePoly(Mat coeffs, Mat roots) {
        return solvePoly_1(coeffs.nativeObj, roots.nativeObj);
    }


    //
    // C++:  float cv::cubeRoot(float val)
    //

    /**
     * Computes the cube root of an argument.
     *
     *  The function cubeRoot computes \(\sqrt[3]{\texttt{val}}\). Negative arguments are handled correctly.
     *  NaN and Inf are not handled. The accuracy approaches the maximum possible accuracy for
     *  single-precision data.
     *  @param val A function argument.
     * @return automatically generated
     */
    public static float cubeRoot(float val) {
        return cubeRoot_0(val);
    }


    //
    // C++:  float cv::fastAtan2(float y, float x)
    //

    /**
     * Calculates the angle of a 2D vector in degrees.
     *
     *  The function fastAtan2 calculates the full-range angle of an input 2D vector. The angle is measured
     *  in degrees and varies from 0 to 360 degrees. The accuracy is about 0.3 degrees.
     *  @param x x-coordinate of the vector.
     *  @param y y-coordinate of the vector.
     * @return automatically generated
     */
    public static float fastAtan2(float y, float x) {
        return fastAtan2_0(y, x);
    }


    //
    // C++:  int cv::borderInterpolate(int p, int len, int borderType)
    //

    /**
     * Computes the source location of an extrapolated pixel.
     *
     * The function computes and returns the coordinate of a donor pixel corresponding to the specified
     * extrapolated pixel when using the specified extrapolation border mode. For example, if you use
     * cv::BORDER_WRAP mode in the horizontal direction, cv::BORDER_REFLECT_101 in the vertical direction and
     * want to compute value of the "virtual" pixel Point(-5, 100) in a floating-point image img , it
     * looks like:
     * <code>
     *     float val = img.at&lt;float&gt;(borderInterpolate(100, img.rows, cv::BORDER_REFLECT_101),
     *                               borderInterpolate(-5, img.cols, cv::BORDER_WRAP));
     * </code>
     * Normally, the function is not called directly. It is used inside filtering functions and also in
     * copyMakeBorder.
     * @param p 0-based coordinate of the extrapolated pixel along one of the axes, likely &lt;0 or &gt;= len
     * @param len Length of the array along the corresponding axis.
     * @param borderType Border type, one of the #BorderTypes, except for #BORDER_TRANSPARENT and
     * #BORDER_ISOLATED . When borderType==#BORDER_CONSTANT , the function always returns -1, regardless
     * of p and len.
     *
     * SEE: copyMakeBorder
     * @return automatically generated
     */
    public static int borderInterpolate(int p, int len, int borderType) {
        return borderInterpolate_0(p, len, borderType);
    }


    //
    // C++:  int cv::countNonZero(Mat src)
    //

    /**
     * Counts non-zero array elements.
     *
     * The function returns the number of non-zero elements in src :
     * \(\sum _{I: \; \texttt{src} (I) \ne0 } 1\)
     * @param src single-channel array.
     * SEE:  mean, meanStdDev, norm, minMaxLoc, calcCovarMatrix
     * @return automatically generated
     */
    public static int countNonZero(Mat src) {
        return countNonZero_0(src.nativeObj);
    }


    //
    // C++:  int cv::getNumThreads()
    //

    /**
     * Returns the number of threads used by OpenCV for parallel regions.
     *
     * Always returns 1 if OpenCV is built without threading support.
     *
     * The exact meaning of return value depends on the threading framework used by OpenCV library:
     * <ul>
     *   <li>
     *  {@code TBB} - The number of threads, that OpenCV will try to use for parallel regions. If there is
     *   any tbb::thread_scheduler_init in user code conflicting with OpenCV, then function returns
     *   default number of threads used by TBB library.
     *   </li>
     *   <li>
     *  {@code OpenMP} - An upper bound on the number of threads that could be used to form a new team.
     *   </li>
     *   <li>
     *  {@code Concurrency} - The number of threads, that OpenCV will try to use for parallel regions.
     *   </li>
     *   <li>
     *  {@code GCD} - Unsupported; returns the GCD thread pool limit (512) for compatibility.
     *   </li>
     *   <li>
     *  {@code C=} - The number of threads, that OpenCV will try to use for parallel regions, if before
     *   called setNumThreads with threads &gt; 0, otherwise returns the number of logical CPUs,
     *   available for the process.
     * SEE: setNumThreads, getThreadNum
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static int getNumThreads() {
        return getNumThreads_0();
    }


    //
    // C++:  int cv::getNumberOfCPUs()
    //

    /**
     * Returns the number of logical CPUs available for the process.
     * @return automatically generated
     */
    public static int getNumberOfCPUs() {
        return getNumberOfCPUs_0();
    }


    //
    // C++:  int cv::getOptimalDFTSize(int vecsize)
    //

    /**
     * Returns the optimal DFT size for a given vector size.
     *
     * DFT performance is not a monotonic function of a vector size. Therefore, when you calculate
     * convolution of two arrays or perform the spectral analysis of an array, it usually makes sense to
     * pad the input data with zeros to get a bit larger array that can be transformed much faster than the
     * original one. Arrays whose size is a power-of-two (2, 4, 8, 16, 32, ...) are the fastest to process.
     * Though, the arrays whose size is a product of 2's, 3's, and 5's (for example, 300 = 5\*5\*3\*2\*2)
     * are also processed quite efficiently.
     *
     * The function cv::getOptimalDFTSize returns the minimum number N that is greater than or equal to vecsize
     * so that the DFT of a vector of size N can be processed efficiently. In the current implementation N
     * = 2 ^p^ \* 3 ^q^ \* 5 ^r^ for some integer p, q, r.
     *
     * The function returns a negative number if vecsize is too large (very close to INT_MAX ).
     *
     * While the function cannot be used directly to estimate the optimal vector size for DCT transform
     * (since the current DCT implementation supports only even-size vectors), it can be easily processed
     * as getOptimalDFTSize((vecsize+1)/2)\*2.
     * @param vecsize vector size.
     * SEE: dft , dct , idft , idct , mulSpectrums
     * @return automatically generated
     */
    public static int getOptimalDFTSize(int vecsize) {
        return getOptimalDFTSize_0(vecsize);
    }


    //
    // C++:  int cv::getThreadNum()
    //

    /**
     * Returns the index of the currently executed thread within the current parallel region. Always
     * returns 0 if called outside of parallel region.
     *
     * @deprecated Current implementation doesn't corresponding to this documentation.
     *
     * The exact meaning of the return value depends on the threading framework used by OpenCV library:
     * <ul>
     *   <li>
     *  {@code TBB} - Unsupported with current 4.1 TBB release. Maybe will be supported in future.
     *   </li>
     *   <li>
     *  {@code OpenMP} - The thread number, within the current team, of the calling thread.
     *   </li>
     *   <li>
     *  {@code Concurrency} - An ID for the virtual processor that the current context is executing on (0
     *   for master thread and unique number for others, but not necessary 1,2,3,...).
     *   </li>
     *   <li>
     *  {@code GCD} - System calling thread's ID. Never returns 0 inside parallel region.
     *   </li>
     *   <li>
     *  {@code C=} - The index of the current parallel task.
     * SEE: setNumThreads, getNumThreads
     *   </li>
     * </ul>
     * @return automatically generated
     */
    @Deprecated
    public static int getThreadNum() {
        return getThreadNum_0();
    }


    //
    // C++:  int cv::getVersionMajor()
    //

    /**
     * Returns major library version
     * @return automatically generated
     */
    public static int getVersionMajor() {
        return getVersionMajor_0();
    }


    //
    // C++:  int cv::getVersionMinor()
    //

    /**
     * Returns minor library version
     * @return automatically generated
     */
    public static int getVersionMinor() {
        return getVersionMinor_0();
    }


    //
    // C++:  int cv::getVersionRevision()
    //

    /**
     * Returns revision field of the library version
     * @return automatically generated
     */
    public static int getVersionRevision() {
        return getVersionRevision_0();
    }


    //
    // C++:  int cv::solveCubic(Mat coeffs, Mat& roots)
    //

    /**
     * Finds the real roots of a cubic equation.
     *
     * The function solveCubic finds the real roots of a cubic equation:
     * <ul>
     *   <li>
     *    if coeffs is a 4-element vector:
     * \(\texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0\)
     *   </li>
     *   <li>
     *    if coeffs is a 3-element vector:
     * \(x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0\)
     *   </li>
     * </ul>
     *
     * The roots are stored in the roots array.
     * @param coeffs equation coefficients, an array of 3 or 4 elements.
     * @param roots output array of real roots that has 1 or 3 elements.
     * @return number of real roots. It can be 0, 1 or 2.
     */
    public static int solveCubic(Mat coeffs, Mat roots) {
        return solveCubic_0(coeffs.nativeObj, roots.nativeObj);
    }


    //
    // C++:  int64 cv::getCPUTickCount()
    //

    /**
     * Returns the number of CPU ticks.
     *
     * The function returns the current number of CPU ticks on some architectures (such as x86, x64,
     * PowerPC). On other platforms the function is equivalent to getTickCount. It can also be used for
     * very accurate time measurements, as well as for RNG initialization. Note that in case of multi-CPU
     * systems a thread, from which getCPUTickCount is called, can be suspended and resumed at another CPU
     * with its own counter. So, theoretically (and practically) the subsequent calls to the function do
     * not necessary return the monotonously increasing values. Also, since a modern CPU varies the CPU
     * frequency depending on the load, the number of CPU clocks spent in some code cannot be directly
     * converted to time units. Therefore, getTickCount is generally a preferable solution for measuring
     * execution time.
     * @return automatically generated
     */
    public static long getCPUTickCount() {
        return getCPUTickCount_0();
    }


    //
    // C++:  int64 cv::getTickCount()
    //

    /**
     * Returns the number of ticks.
     *
     * The function returns the number of ticks after the certain event (for example, when the machine was
     * turned on). It can be used to initialize RNG or to measure a function execution time by reading the
     * tick count before and after the function call.
     * SEE: getTickFrequency, TickMeter
     * @return automatically generated
     */
    public static long getTickCount() {
        return getTickCount_0();
    }


    //
    // C++:  string cv::getCPUFeaturesLine()
    //

    /**
     * Returns list of CPU features enabled during compilation.
     *
     * Returned value is a string containing space separated list of CPU features with following markers:
     *
     * <ul>
     *   <li>
     *  no markers - baseline features
     *   </li>
     *   <li>
     *  prefix {@code *} - features enabled in dispatcher
     *   </li>
     *   <li>
     *  suffix {@code ?} - features enabled but not available in HW
     *   </li>
     * </ul>
     *
     * Example: {@code SSE SSE2 SSE3 *SSE4.1 *SSE4.2 *FP16 *AVX *AVX2 *AVX512-SKX?}
     * @return automatically generated
     */
    public static String getCPUFeaturesLine() {
        return getCPUFeaturesLine_0();
    }


    //
    // C++:  void cv::LUT(Mat src, Mat lut, Mat& dst)
    //

    /**
     * Performs a look-up table transform of an array.
     *
     * The function LUT fills the output array with values from the look-up table. Indices of the entries
     * are taken from the input array. That is, the function processes each element of src as follows:
     * \(\texttt{dst} (I)  \leftarrow \texttt{lut(src(I) + d)}\)
     * where
     * \(d =  \fork{0}{if \(\texttt{src}\) has depth \(\texttt{CV_8U}\)}{128}{if \(\texttt{src}\) has depth \(\texttt{CV_8S}\)}\)
     * @param src input array of 8-bit elements.
     * @param lut look-up table of 256 elements; in case of multi-channel input array, the table should
     * either have a single channel (in this case the same table is used for all channels) or the same
     * number of channels as in the input array.
     * @param dst output array of the same size and number of channels as src, and the same depth as lut.
     * SEE:  convertScaleAbs, Mat::convertTo
     */
    public static void LUT(Mat src, Mat lut, Mat dst) {
        LUT_0(src.nativeObj, lut.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::PCABackProject(Mat data, Mat mean, Mat eigenvectors, Mat& result)
    //

    /**
     * wrap PCA::backProject
     * @param data automatically generated
     * @param mean automatically generated
     * @param eigenvectors automatically generated
     * @param result automatically generated
     */
    public static void PCABackProject(Mat data, Mat mean, Mat eigenvectors, Mat result) {
        PCABackProject_0(data.nativeObj, mean.nativeObj, eigenvectors.nativeObj, result.nativeObj);
    }


    //
    // C++:  void cv::PCACompute(Mat data, Mat& mean, Mat& eigenvectors, Mat& eigenvalues, double retainedVariance)
    //

    /**
     * wrap PCA::operator() and add eigenvalues output parameter
     * @param data automatically generated
     * @param mean automatically generated
     * @param eigenvectors automatically generated
     * @param eigenvalues automatically generated
     * @param retainedVariance automatically generated
     */
    public static void PCACompute2(Mat data, Mat mean, Mat eigenvectors, Mat eigenvalues, double retainedVariance) {
        PCACompute2_0(data.nativeObj, mean.nativeObj, eigenvectors.nativeObj, eigenvalues.nativeObj, retainedVariance);
    }


    //
    // C++:  void cv::PCACompute(Mat data, Mat& mean, Mat& eigenvectors, Mat& eigenvalues, int maxComponents = 0)
    //

    /**
     * wrap PCA::operator() and add eigenvalues output parameter
     * @param data automatically generated
     * @param mean automatically generated
     * @param eigenvectors automatically generated
     * @param eigenvalues automatically generated
     * @param maxComponents automatically generated
     */
    public static void PCACompute2(Mat data, Mat mean, Mat eigenvectors, Mat eigenvalues, int maxComponents) {
        PCACompute2_1(data.nativeObj, mean.nativeObj, eigenvectors.nativeObj, eigenvalues.nativeObj, maxComponents);
    }

    /**
     * wrap PCA::operator() and add eigenvalues output parameter
     * @param data automatically generated
     * @param mean automatically generated
     * @param eigenvectors automatically generated
     * @param eigenvalues automatically generated
     */
    public static void PCACompute2(Mat data, Mat mean, Mat eigenvectors, Mat eigenvalues) {
        PCACompute2_2(data.nativeObj, mean.nativeObj, eigenvectors.nativeObj, eigenvalues.nativeObj);
    }


    //
    // C++:  void cv::PCACompute(Mat data, Mat& mean, Mat& eigenvectors, double retainedVariance)
    //

    /**
     * wrap PCA::operator()
     * @param data automatically generated
     * @param mean automatically generated
     * @param eigenvectors automatically generated
     * @param retainedVariance automatically generated
     */
    public static void PCACompute(Mat data, Mat mean, Mat eigenvectors, double retainedVariance) {
        PCACompute_0(data.nativeObj, mean.nativeObj, eigenvectors.nativeObj, retainedVariance);
    }


    //
    // C++:  void cv::PCACompute(Mat data, Mat& mean, Mat& eigenvectors, int maxComponents = 0)
    //

    /**
     * wrap PCA::operator()
     * @param data automatically generated
     * @param mean automatically generated
     * @param eigenvectors automatically generated
     * @param maxComponents automatically generated
     */
    public static void PCACompute(Mat data, Mat mean, Mat eigenvectors, int maxComponents) {
        PCACompute_1(data.nativeObj, mean.nativeObj, eigenvectors.nativeObj, maxComponents);
    }

    /**
     * wrap PCA::operator()
     * @param data automatically generated
     * @param mean automatically generated
     * @param eigenvectors automatically generated
     */
    public static void PCACompute(Mat data, Mat mean, Mat eigenvectors) {
        PCACompute_2(data.nativeObj, mean.nativeObj, eigenvectors.nativeObj);
    }


    //
    // C++:  void cv::PCAProject(Mat data, Mat mean, Mat eigenvectors, Mat& result)
    //

    /**
     * wrap PCA::project
     * @param data automatically generated
     * @param mean automatically generated
     * @param eigenvectors automatically generated
     * @param result automatically generated
     */
    public static void PCAProject(Mat data, Mat mean, Mat eigenvectors, Mat result) {
        PCAProject_0(data.nativeObj, mean.nativeObj, eigenvectors.nativeObj, result.nativeObj);
    }


    //
    // C++:  void cv::SVBackSubst(Mat w, Mat u, Mat vt, Mat rhs, Mat& dst)
    //

    /**
     * wrap SVD::backSubst
     * @param w automatically generated
     * @param u automatically generated
     * @param vt automatically generated
     * @param rhs automatically generated
     * @param dst automatically generated
     */
    public static void SVBackSubst(Mat w, Mat u, Mat vt, Mat rhs, Mat dst) {
        SVBackSubst_0(w.nativeObj, u.nativeObj, vt.nativeObj, rhs.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::SVDecomp(Mat src, Mat& w, Mat& u, Mat& vt, int flags = 0)
    //

    /**
     * wrap SVD::compute
     * @param src automatically generated
     * @param w automatically generated
     * @param u automatically generated
     * @param vt automatically generated
     * @param flags automatically generated
     */
    public static void SVDecomp(Mat src, Mat w, Mat u, Mat vt, int flags) {
        SVDecomp_0(src.nativeObj, w.nativeObj, u.nativeObj, vt.nativeObj, flags);
    }

    /**
     * wrap SVD::compute
     * @param src automatically generated
     * @param w automatically generated
     * @param u automatically generated
     * @param vt automatically generated
     */
    public static void SVDecomp(Mat src, Mat w, Mat u, Mat vt) {
        SVDecomp_1(src.nativeObj, w.nativeObj, u.nativeObj, vt.nativeObj);
    }


    //
    // C++:  void cv::absdiff(Mat src1, Mat src2, Mat& dst)
    //

    /**
     * Calculates the per-element absolute difference between two arrays or between an array and a scalar.
     *
     * The function cv::absdiff calculates:
     * Absolute difference between two arrays when they have the same
     *     size and type:
     *     \(\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)\)
     * Absolute difference between an array and a scalar when the second
     *     array is constructed from Scalar or has as many elements as the
     *     number of channels in {@code src1}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2} |)\)
     * Absolute difference between a scalar and an array when the first
     *     array is constructed from Scalar or has as many elements as the
     *     number of channels in {@code src2}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1} -  \texttt{src2}(I) |)\)
     *     where I is a multi-dimensional index of array elements. In case of
     *     multi-channel arrays, each channel is processed independently.
     * <b>Note:</b> Saturation is not applied when the arrays have the depth CV_32S.
     * You may even get a negative value in the case of overflow.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and type as input arrays.
     * SEE: cv::abs(const Mat&amp;)
     */
    public static void absdiff(Mat src1, Mat src2, Mat dst) {
        absdiff_0(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::absdiff(Mat src1, Scalar src2, Mat& dst)
    //

    public static void absdiff(Mat src1, Scalar src2, Mat dst) {
        absdiff_1(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj);
    }


    //
    // C++:  void cv::add(Mat src1, Mat src2, Mat& dst, Mat mask = Mat(), int dtype = -1)
    //

    /**
     * Calculates the per-element sum of two arrays or an array and a scalar.
     *
     * The function add calculates:
     * <ul>
     *   <li>
     *  Sum of two arrays when both input arrays have the same size and the same number of channels:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of
     * elements as {@code src1.channels()}:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of
     * elements as {@code src2.channels()}:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\)
     * where {@code I} is a multi-dimensional index of array elements. In case of multi-channel arrays, each
     * channel is processed independently.
     *   </li>
     * </ul>
     *
     * The first function in the list above can be replaced with matrix expressions:
     * <code>
     *     dst = src1 + src2;
     *     dst += src1; // equivalent to add(dst, src1, dst);
     * </code>
     * The input arrays and the output array can all have the same or different depths. For example, you
     * can add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit
     * floating-point array. Depth of the output array is determined by the dtype parameter. In the second
     * and third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can
     * be set to the default -1. In this case, the output array will have the same depth as the input
     * array, be it src1, src2 or both.
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and number of channels as the input array(s); the
     * depth is defined by dtype or src1/src2.
     * @param mask optional operation mask - 8-bit single channel array, that specifies elements of the
     * output array to be changed.
     * @param dtype optional depth of the output array (see the discussion below).
     * SEE: subtract, addWeighted, scaleAdd, Mat::convertTo
     */
    public static void add(Mat src1, Mat src2, Mat dst, Mat mask, int dtype) {
        add_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj, dtype);
    }

    /**
     * Calculates the per-element sum of two arrays or an array and a scalar.
     *
     * The function add calculates:
     * <ul>
     *   <li>
     *  Sum of two arrays when both input arrays have the same size and the same number of channels:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of
     * elements as {@code src1.channels()}:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of
     * elements as {@code src2.channels()}:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\)
     * where {@code I} is a multi-dimensional index of array elements. In case of multi-channel arrays, each
     * channel is processed independently.
     *   </li>
     * </ul>
     *
     * The first function in the list above can be replaced with matrix expressions:
     * <code>
     *     dst = src1 + src2;
     *     dst += src1; // equivalent to add(dst, src1, dst);
     * </code>
     * The input arrays and the output array can all have the same or different depths. For example, you
     * can add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit
     * floating-point array. Depth of the output array is determined by the dtype parameter. In the second
     * and third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can
     * be set to the default -1. In this case, the output array will have the same depth as the input
     * array, be it src1, src2 or both.
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and number of channels as the input array(s); the
     * depth is defined by dtype or src1/src2.
     * @param mask optional operation mask - 8-bit single channel array, that specifies elements of the
     * output array to be changed.
     * SEE: subtract, addWeighted, scaleAdd, Mat::convertTo
     */
    public static void add(Mat src1, Mat src2, Mat dst, Mat mask) {
        add_1(src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     * Calculates the per-element sum of two arrays or an array and a scalar.
     *
     * The function add calculates:
     * <ul>
     *   <li>
     *  Sum of two arrays when both input arrays have the same size and the same number of channels:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of
     * elements as {@code src1.channels()}:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of
     * elements as {@code src2.channels()}:
     * \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\)
     * where {@code I} is a multi-dimensional index of array elements. In case of multi-channel arrays, each
     * channel is processed independently.
     *   </li>
     * </ul>
     *
     * The first function in the list above can be replaced with matrix expressions:
     * <code>
     *     dst = src1 + src2;
     *     dst += src1; // equivalent to add(dst, src1, dst);
     * </code>
     * The input arrays and the output array can all have the same or different depths. For example, you
     * can add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit
     * floating-point array. Depth of the output array is determined by the dtype parameter. In the second
     * and third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can
     * be set to the default -1. In this case, the output array will have the same depth as the input
     * array, be it src1, src2 or both.
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and number of channels as the input array(s); the
     * depth is defined by dtype or src1/src2.
     * output array to be changed.
     * SEE: subtract, addWeighted, scaleAdd, Mat::convertTo
     */
    public static void add(Mat src1, Mat src2, Mat dst) {
        add_2(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::add(Mat src1, Scalar src2, Mat& dst, Mat mask = Mat(), int dtype = -1)
    //

    public static void add(Mat src1, Scalar src2, Mat dst, Mat mask, int dtype) {
        add_3(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, mask.nativeObj, dtype);
    }

    public static void add(Mat src1, Scalar src2, Mat dst, Mat mask) {
        add_4(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, mask.nativeObj);
    }

    public static void add(Mat src1, Scalar src2, Mat dst) {
        add_5(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj);
    }


    //
    // C++:  void cv::addWeighted(Mat src1, double alpha, Mat src2, double beta, double gamma, Mat& dst, int dtype = -1)
    //

    /**
     * Calculates the weighted sum of two arrays.
     *
     * The function addWeighted calculates the weighted sum of two arrays as follows:
     * \(\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )\)
     * where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
     * channel is processed independently.
     * The function can be replaced with a matrix expression:
     * <code>
     *     dst = src1*alpha + src2*beta + gamma;
     * </code>
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array.
     * @param alpha weight of the first array elements.
     * @param src2 second input array of the same size and channel number as src1.
     * @param beta weight of the second array elements.
     * @param gamma scalar added to each sum.
     * @param dst output array that has the same size and number of channels as the input arrays.
     * @param dtype optional depth of the output array; when both input arrays have the same depth, dtype
     * can be set to -1, which will be equivalent to src1.depth().
     * SEE:  add, subtract, scaleAdd, Mat::convertTo
     */
    public static void addWeighted(Mat src1, double alpha, Mat src2, double beta, double gamma, Mat dst, int dtype) {
        addWeighted_0(src1.nativeObj, alpha, src2.nativeObj, beta, gamma, dst.nativeObj, dtype);
    }

    /**
     * Calculates the weighted sum of two arrays.
     *
     * The function addWeighted calculates the weighted sum of two arrays as follows:
     * \(\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )\)
     * where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
     * channel is processed independently.
     * The function can be replaced with a matrix expression:
     * <code>
     *     dst = src1*alpha + src2*beta + gamma;
     * </code>
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array.
     * @param alpha weight of the first array elements.
     * @param src2 second input array of the same size and channel number as src1.
     * @param beta weight of the second array elements.
     * @param gamma scalar added to each sum.
     * @param dst output array that has the same size and number of channels as the input arrays.
     * can be set to -1, which will be equivalent to src1.depth().
     * SEE:  add, subtract, scaleAdd, Mat::convertTo
     */
    public static void addWeighted(Mat src1, double alpha, Mat src2, double beta, double gamma, Mat dst) {
        addWeighted_1(src1.nativeObj, alpha, src2.nativeObj, beta, gamma, dst.nativeObj);
    }


    //
    // C++:  void cv::batchDistance(Mat src1, Mat src2, Mat& dist, int dtype, Mat& nidx, int normType = NORM_L2, int K = 0, Mat mask = Mat(), int update = 0, bool crosscheck = false)
    //

    /**
     * naive nearest neighbor finder
     *
     * see http://en.wikipedia.org/wiki/Nearest_neighbor_search
     * TODO: document
     * @param src1 automatically generated
     * @param src2 automatically generated
     * @param dist automatically generated
     * @param dtype automatically generated
     * @param nidx automatically generated
     * @param normType automatically generated
     * @param K automatically generated
     * @param mask automatically generated
     * @param update automatically generated
     * @param crosscheck automatically generated
     */
    public static void batchDistance(Mat src1, Mat src2, Mat dist, int dtype, Mat nidx, int normType, int K, Mat mask, int update, boolean crosscheck) {
        batchDistance_0(src1.nativeObj, src2.nativeObj, dist.nativeObj, dtype, nidx.nativeObj, normType, K, mask.nativeObj, update, crosscheck);
    }

    /**
     * naive nearest neighbor finder
     *
     * see http://en.wikipedia.org/wiki/Nearest_neighbor_search
     * TODO: document
     * @param src1 automatically generated
     * @param src2 automatically generated
     * @param dist automatically generated
     * @param dtype automatically generated
     * @param nidx automatically generated
     * @param normType automatically generated
     * @param K automatically generated
     * @param mask automatically generated
     * @param update automatically generated
     */
    public static void batchDistance(Mat src1, Mat src2, Mat dist, int dtype, Mat nidx, int normType, int K, Mat mask, int update) {
        batchDistance_1(src1.nativeObj, src2.nativeObj, dist.nativeObj, dtype, nidx.nativeObj, normType, K, mask.nativeObj, update);
    }

    /**
     * naive nearest neighbor finder
     *
     * see http://en.wikipedia.org/wiki/Nearest_neighbor_search
     * TODO: document
     * @param src1 automatically generated
     * @param src2 automatically generated
     * @param dist automatically generated
     * @param dtype automatically generated
     * @param nidx automatically generated
     * @param normType automatically generated
     * @param K automatically generated
     * @param mask automatically generated
     */
    public static void batchDistance(Mat src1, Mat src2, Mat dist, int dtype, Mat nidx, int normType, int K, Mat mask) {
        batchDistance_2(src1.nativeObj, src2.nativeObj, dist.nativeObj, dtype, nidx.nativeObj, normType, K, mask.nativeObj);
    }

    /**
     * naive nearest neighbor finder
     *
     * see http://en.wikipedia.org/wiki/Nearest_neighbor_search
     * TODO: document
     * @param src1 automatically generated
     * @param src2 automatically generated
     * @param dist automatically generated
     * @param dtype automatically generated
     * @param nidx automatically generated
     * @param normType automatically generated
     * @param K automatically generated
     */
    public static void batchDistance(Mat src1, Mat src2, Mat dist, int dtype, Mat nidx, int normType, int K) {
        batchDistance_3(src1.nativeObj, src2.nativeObj, dist.nativeObj, dtype, nidx.nativeObj, normType, K);
    }

    /**
     * naive nearest neighbor finder
     *
     * see http://en.wikipedia.org/wiki/Nearest_neighbor_search
     * TODO: document
     * @param src1 automatically generated
     * @param src2 automatically generated
     * @param dist automatically generated
     * @param dtype automatically generated
     * @param nidx automatically generated
     * @param normType automatically generated
     */
    public static void batchDistance(Mat src1, Mat src2, Mat dist, int dtype, Mat nidx, int normType) {
        batchDistance_4(src1.nativeObj, src2.nativeObj, dist.nativeObj, dtype, nidx.nativeObj, normType);
    }

    /**
     * naive nearest neighbor finder
     *
     * see http://en.wikipedia.org/wiki/Nearest_neighbor_search
     * TODO: document
     * @param src1 automatically generated
     * @param src2 automatically generated
     * @param dist automatically generated
     * @param dtype automatically generated
     * @param nidx automatically generated
     */
    public static void batchDistance(Mat src1, Mat src2, Mat dist, int dtype, Mat nidx) {
        batchDistance_5(src1.nativeObj, src2.nativeObj, dist.nativeObj, dtype, nidx.nativeObj);
    }


    //
    // C++:  void cv::bitwise_and(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
    //

    /**
     * computes bitwise conjunction of the two arrays (dst = src1 &amp; src2)
     * Calculates the per-element bit-wise conjunction of two arrays or an
     * array and a scalar.
     *
     * The function cv::bitwise_and calculates the per-element bit-wise logical conjunction for:
     * Two arrays when src1 and src2 have the same size:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * An array and a scalar when src2 is constructed from Scalar or has
     *     the same number of elements as {@code src1.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0\)
     * A scalar and an array when src1 is constructed from Scalar or has
     *     the same number of elements as {@code src2.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * In case of floating-point arrays, their machine-specific bit
     * representations (usually IEEE754-compliant) are used for the operation.
     * In case of multi-channel arrays, each channel is processed
     * independently. In the second and third cases above, the scalar is first
     * converted to the array type.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and type as the input
     * arrays.
     * @param mask optional operation mask, 8-bit single channel array, that
     * specifies elements of the output array to be changed.
     */
    public static void bitwise_and(Mat src1, Mat src2, Mat dst, Mat mask) {
        bitwise_and_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     * computes bitwise conjunction of the two arrays (dst = src1 &amp; src2)
     * Calculates the per-element bit-wise conjunction of two arrays or an
     * array and a scalar.
     *
     * The function cv::bitwise_and calculates the per-element bit-wise logical conjunction for:
     * Two arrays when src1 and src2 have the same size:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * An array and a scalar when src2 is constructed from Scalar or has
     *     the same number of elements as {@code src1.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0\)
     * A scalar and an array when src1 is constructed from Scalar or has
     *     the same number of elements as {@code src2.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * In case of floating-point arrays, their machine-specific bit
     * representations (usually IEEE754-compliant) are used for the operation.
     * In case of multi-channel arrays, each channel is processed
     * independently. In the second and third cases above, the scalar is first
     * converted to the array type.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and type as the input
     * arrays.
     * specifies elements of the output array to be changed.
     */
    public static void bitwise_and(Mat src1, Mat src2, Mat dst) {
        bitwise_and_1(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::bitwise_not(Mat src, Mat& dst, Mat mask = Mat())
    //

    /**
     *  Inverts every bit of an array.
     *
     * The function cv::bitwise_not calculates per-element bit-wise inversion of the input
     * array:
     * \(\texttt{dst} (I) =  \neg \texttt{src} (I)\)
     * In case of a floating-point input array, its machine-specific bit
     * representation (usually IEEE754-compliant) is used for the operation. In
     * case of multi-channel arrays, each channel is processed independently.
     * @param src input array.
     * @param dst output array that has the same size and type as the input
     * array.
     * @param mask optional operation mask, 8-bit single channel array, that
     * specifies elements of the output array to be changed.
     */
    public static void bitwise_not(Mat src, Mat dst, Mat mask) {
        bitwise_not_0(src.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     *  Inverts every bit of an array.
     *
     * The function cv::bitwise_not calculates per-element bit-wise inversion of the input
     * array:
     * \(\texttt{dst} (I) =  \neg \texttt{src} (I)\)
     * In case of a floating-point input array, its machine-specific bit
     * representation (usually IEEE754-compliant) is used for the operation. In
     * case of multi-channel arrays, each channel is processed independently.
     * @param src input array.
     * @param dst output array that has the same size and type as the input
     * array.
     * specifies elements of the output array to be changed.
     */
    public static void bitwise_not(Mat src, Mat dst) {
        bitwise_not_1(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::bitwise_or(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
    //

    /**
     * Calculates the per-element bit-wise disjunction of two arrays or an
     * array and a scalar.
     *
     * The function cv::bitwise_or calculates the per-element bit-wise logical disjunction for:
     * Two arrays when src1 and src2 have the same size:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * An array and a scalar when src2 is constructed from Scalar or has
     *     the same number of elements as {@code src1.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0\)
     * A scalar and an array when src1 is constructed from Scalar or has
     *     the same number of elements as {@code src2.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * In case of floating-point arrays, their machine-specific bit
     * representations (usually IEEE754-compliant) are used for the operation.
     * In case of multi-channel arrays, each channel is processed
     * independently. In the second and third cases above, the scalar is first
     * converted to the array type.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and type as the input
     * arrays.
     * @param mask optional operation mask, 8-bit single channel array, that
     * specifies elements of the output array to be changed.
     */
    public static void bitwise_or(Mat src1, Mat src2, Mat dst, Mat mask) {
        bitwise_or_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     * Calculates the per-element bit-wise disjunction of two arrays or an
     * array and a scalar.
     *
     * The function cv::bitwise_or calculates the per-element bit-wise logical disjunction for:
     * Two arrays when src1 and src2 have the same size:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * An array and a scalar when src2 is constructed from Scalar or has
     *     the same number of elements as {@code src1.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0\)
     * A scalar and an array when src1 is constructed from Scalar or has
     *     the same number of elements as {@code src2.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * In case of floating-point arrays, their machine-specific bit
     * representations (usually IEEE754-compliant) are used for the operation.
     * In case of multi-channel arrays, each channel is processed
     * independently. In the second and third cases above, the scalar is first
     * converted to the array type.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and type as the input
     * arrays.
     * specifies elements of the output array to be changed.
     */
    public static void bitwise_or(Mat src1, Mat src2, Mat dst) {
        bitwise_or_1(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::bitwise_xor(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
    //

    /**
     * Calculates the per-element bit-wise "exclusive or" operation on two
     * arrays or an array and a scalar.
     *
     * The function cv::bitwise_xor calculates the per-element bit-wise logical "exclusive-or"
     * operation for:
     * Two arrays when src1 and src2 have the same size:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * An array and a scalar when src2 is constructed from Scalar or has
     *     the same number of elements as {@code src1.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0\)
     * A scalar and an array when src1 is constructed from Scalar or has
     *     the same number of elements as {@code src2.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * In case of floating-point arrays, their machine-specific bit
     * representations (usually IEEE754-compliant) are used for the operation.
     * In case of multi-channel arrays, each channel is processed
     * independently. In the 2nd and 3rd cases above, the scalar is first
     * converted to the array type.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and type as the input
     * arrays.
     * @param mask optional operation mask, 8-bit single channel array, that
     * specifies elements of the output array to be changed.
     */
    public static void bitwise_xor(Mat src1, Mat src2, Mat dst, Mat mask) {
        bitwise_xor_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     * Calculates the per-element bit-wise "exclusive or" operation on two
     * arrays or an array and a scalar.
     *
     * The function cv::bitwise_xor calculates the per-element bit-wise logical "exclusive-or"
     * operation for:
     * Two arrays when src1 and src2 have the same size:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * An array and a scalar when src2 is constructed from Scalar or has
     *     the same number of elements as {@code src1.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0\)
     * A scalar and an array when src1 is constructed from Scalar or has
     *     the same number of elements as {@code src2.channels()}:
     *     \(\texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\)
     * In case of floating-point arrays, their machine-specific bit
     * representations (usually IEEE754-compliant) are used for the operation.
     * In case of multi-channel arrays, each channel is processed
     * independently. In the 2nd and 3rd cases above, the scalar is first
     * converted to the array type.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array that has the same size and type as the input
     * arrays.
     * specifies elements of the output array to be changed.
     */
    public static void bitwise_xor(Mat src1, Mat src2, Mat dst) {
        bitwise_xor_1(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::calcCovarMatrix(Mat samples, Mat& covar, Mat& mean, int flags, int ctype = CV_64F)
    //

    /**
     *
     * <b>Note:</b> use #COVAR_ROWS or #COVAR_COLS flag
     * @param samples samples stored as rows/columns of a single matrix.
     * @param covar output covariance matrix of the type ctype and square size.
     * @param mean input or output (depending on the flags) array as the average value of the input vectors.
     * @param flags operation flags as a combination of #CovarFlags
     * @param ctype type of the matrixl; it equals 'CV_64F' by default.
     */
    public static void calcCovarMatrix(Mat samples, Mat covar, Mat mean, int flags, int ctype) {
        calcCovarMatrix_0(samples.nativeObj, covar.nativeObj, mean.nativeObj, flags, ctype);
    }

    /**
     *
     * <b>Note:</b> use #COVAR_ROWS or #COVAR_COLS flag
     * @param samples samples stored as rows/columns of a single matrix.
     * @param covar output covariance matrix of the type ctype and square size.
     * @param mean input or output (depending on the flags) array as the average value of the input vectors.
     * @param flags operation flags as a combination of #CovarFlags
     */
    public static void calcCovarMatrix(Mat samples, Mat covar, Mat mean, int flags) {
        calcCovarMatrix_1(samples.nativeObj, covar.nativeObj, mean.nativeObj, flags);
    }


    //
    // C++:  void cv::cartToPolar(Mat x, Mat y, Mat& magnitude, Mat& angle, bool angleInDegrees = false)
    //

    /**
     * Calculates the magnitude and angle of 2D vectors.
     *
     * The function cv::cartToPolar calculates either the magnitude, angle, or both
     * for every 2D vector (x(I),y(I)):
     * \(\begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2} , \\ \texttt{angle} (I)= \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))[ \cdot180 / \pi ] \end{array}\)
     *
     * The angles are calculated with accuracy about 0.3 degrees. For the point
     * (0,0), the angle is set to 0.
     * @param x array of x-coordinates; this must be a single-precision or
     * double-precision floating-point array.
     * @param y array of y-coordinates, that must have the same size and same type as x.
     * @param magnitude output array of magnitudes of the same size and type as x.
     * @param angle output array of angles that has the same size and type as
     * x; the angles are measured in radians (from 0 to 2\*Pi) or in degrees (0 to 360 degrees).
     * @param angleInDegrees a flag, indicating whether the angles are measured
     * in radians (which is by default), or in degrees.
     * SEE: Sobel, Scharr
     */
    public static void cartToPolar(Mat x, Mat y, Mat magnitude, Mat angle, boolean angleInDegrees) {
        cartToPolar_0(x.nativeObj, y.nativeObj, magnitude.nativeObj, angle.nativeObj, angleInDegrees);
    }

    /**
     * Calculates the magnitude and angle of 2D vectors.
     *
     * The function cv::cartToPolar calculates either the magnitude, angle, or both
     * for every 2D vector (x(I),y(I)):
     * \(\begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2} , \\ \texttt{angle} (I)= \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))[ \cdot180 / \pi ] \end{array}\)
     *
     * The angles are calculated with accuracy about 0.3 degrees. For the point
     * (0,0), the angle is set to 0.
     * @param x array of x-coordinates; this must be a single-precision or
     * double-precision floating-point array.
     * @param y array of y-coordinates, that must have the same size and same type as x.
     * @param magnitude output array of magnitudes of the same size and type as x.
     * @param angle output array of angles that has the same size and type as
     * x; the angles are measured in radians (from 0 to 2\*Pi) or in degrees (0 to 360 degrees).
     * in radians (which is by default), or in degrees.
     * SEE: Sobel, Scharr
     */
    public static void cartToPolar(Mat x, Mat y, Mat magnitude, Mat angle) {
        cartToPolar_1(x.nativeObj, y.nativeObj, magnitude.nativeObj, angle.nativeObj);
    }


    //
    // C++:  void cv::compare(Mat src1, Mat src2, Mat& dst, int cmpop)
    //

    /**
     * Performs the per-element comparison of two arrays or an array and scalar value.
     *
     * The function compares:
     * Elements of two arrays when src1 and src2 have the same size:
     *     \(\texttt{dst} (I) =  \texttt{src1} (I)  \,\texttt{cmpop}\, \texttt{src2} (I)\)
     * Elements of src1 with a scalar src2 when src2 is constructed from
     *     Scalar or has a single element:
     *     \(\texttt{dst} (I) =  \texttt{src1}(I) \,\texttt{cmpop}\,  \texttt{src2}\)
     * src1 with elements of src2 when src1 is constructed from Scalar or
     *     has a single element:
     *     \(\texttt{dst} (I) =  \texttt{src1}  \,\texttt{cmpop}\, \texttt{src2} (I)\)
     * When the comparison result is true, the corresponding element of output
     * array is set to 255. The comparison operations can be replaced with the
     * equivalent matrix expressions:
     * <code>
     *     Mat dst1 = src1 &gt;= src2;
     *     Mat dst2 = src1 &lt; 8;
     *     ...
     * </code>
     * @param src1 first input array or a scalar; when it is an array, it must have a single channel.
     * @param src2 second input array or a scalar; when it is an array, it must have a single channel.
     * @param dst output array of type ref CV_8U that has the same size and the same number of channels as
     *     the input arrays.
     * @param cmpop a flag, that specifies correspondence between the arrays (cv::CmpTypes)
     * SEE: checkRange, min, max, threshold
     */
    public static void compare(Mat src1, Mat src2, Mat dst, int cmpop) {
        compare_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, cmpop);
    }


    //
    // C++:  void cv::compare(Mat src1, Scalar src2, Mat& dst, int cmpop)
    //

    public static void compare(Mat src1, Scalar src2, Mat dst, int cmpop) {
        compare_1(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, cmpop);
    }


    //
    // C++:  void cv::completeSymm(Mat& m, bool lowerToUpper = false)
    //

    /**
     * Copies the lower or the upper half of a square matrix to its another half.
     *
     * The function cv::completeSymm copies the lower or the upper half of a square matrix to
     * its another half. The matrix diagonal remains unchanged:
     * <ul>
     *   <li>
     *   \(\texttt{m}_{ij}=\texttt{m}_{ji}\) for \(i &gt; j\) if
     *     lowerToUpper=false
     *   </li>
     *   <li>
     *   \(\texttt{m}_{ij}=\texttt{m}_{ji}\) for \(i &lt; j\) if
     *     lowerToUpper=true
     *   </li>
     * </ul>
     *
     * @param m input-output floating-point square matrix.
     * @param lowerToUpper operation flag; if true, the lower half is copied to
     * the upper half. Otherwise, the upper half is copied to the lower half.
     * SEE: flip, transpose
     */
    public static void completeSymm(Mat m, boolean lowerToUpper) {
        completeSymm_0(m.nativeObj, lowerToUpper);
    }

    /**
     * Copies the lower or the upper half of a square matrix to its another half.
     *
     * The function cv::completeSymm copies the lower or the upper half of a square matrix to
     * its another half. The matrix diagonal remains unchanged:
     * <ul>
     *   <li>
     *   \(\texttt{m}_{ij}=\texttt{m}_{ji}\) for \(i &gt; j\) if
     *     lowerToUpper=false
     *   </li>
     *   <li>
     *   \(\texttt{m}_{ij}=\texttt{m}_{ji}\) for \(i &lt; j\) if
     *     lowerToUpper=true
     *   </li>
     * </ul>
     *
     * @param m input-output floating-point square matrix.
     * the upper half. Otherwise, the upper half is copied to the lower half.
     * SEE: flip, transpose
     */
    public static void completeSymm(Mat m) {
        completeSymm_1(m.nativeObj);
    }


    //
    // C++:  void cv::convertFp16(Mat src, Mat& dst)
    //

    /**
     * Converts an array to half precision floating number.
     *
     * This function converts FP32 (single precision floating point) from/to FP16 (half precision floating point). CV_16S format is used to represent FP16 data.
     * There are two use modes (src -&gt; dst): CV_32F -&gt; CV_16S and CV_16S -&gt; CV_32F. The input array has to have type of CV_32F or
     * CV_16S to represent the bit depth. If the input array is neither of them, the function will raise an error.
     * The format of half precision floating point is defined in IEEE 754-2008.
     *
     * @param src input array.
     * @param dst output array.
     */
    public static void convertFp16(Mat src, Mat dst) {
        convertFp16_0(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::convertScaleAbs(Mat src, Mat& dst, double alpha = 1, double beta = 0)
    //

    /**
     * Scales, calculates absolute values, and converts the result to 8-bit.
     *
     * On each element of the input array, the function convertScaleAbs
     * performs three operations sequentially: scaling, taking an absolute
     * value, conversion to an unsigned 8-bit type:
     * \(\texttt{dst} (I)= \texttt{saturate\_cast&lt;uchar&gt;} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\)
     * In case of multi-channel arrays, the function processes each channel
     * independently. When the output is not 8-bit, the operation can be
     * emulated by calling the Mat::convertTo method (or by using matrix
     * expressions) and then by calculating an absolute value of the result.
     * For example:
     * <code>
     *     Mat_&lt;float&gt; A(30,30);
     *     randu(A, Scalar(-100), Scalar(100));
     *     Mat_&lt;float&gt; B = A*5 + 3;
     *     B = abs(B);
     *     // Mat_&lt;float&gt; B = abs(A*5+3) will also do the job,
     *     // but it will allocate a temporary matrix
     * </code>
     * @param src input array.
     * @param dst output array.
     * @param alpha optional scale factor.
     * @param beta optional delta added to the scaled values.
     * SEE:  Mat::convertTo, cv::abs(const Mat&amp;)
     */
    public static void convertScaleAbs(Mat src, Mat dst, double alpha, double beta) {
        convertScaleAbs_0(src.nativeObj, dst.nativeObj, alpha, beta);
    }

    /**
     * Scales, calculates absolute values, and converts the result to 8-bit.
     *
     * On each element of the input array, the function convertScaleAbs
     * performs three operations sequentially: scaling, taking an absolute
     * value, conversion to an unsigned 8-bit type:
     * \(\texttt{dst} (I)= \texttt{saturate\_cast&lt;uchar&gt;} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\)
     * In case of multi-channel arrays, the function processes each channel
     * independently. When the output is not 8-bit, the operation can be
     * emulated by calling the Mat::convertTo method (or by using matrix
     * expressions) and then by calculating an absolute value of the result.
     * For example:
     * <code>
     *     Mat_&lt;float&gt; A(30,30);
     *     randu(A, Scalar(-100), Scalar(100));
     *     Mat_&lt;float&gt; B = A*5 + 3;
     *     B = abs(B);
     *     // Mat_&lt;float&gt; B = abs(A*5+3) will also do the job,
     *     // but it will allocate a temporary matrix
     * </code>
     * @param src input array.
     * @param dst output array.
     * @param alpha optional scale factor.
     * SEE:  Mat::convertTo, cv::abs(const Mat&amp;)
     */
    public static void convertScaleAbs(Mat src, Mat dst, double alpha) {
        convertScaleAbs_1(src.nativeObj, dst.nativeObj, alpha);
    }

    /**
     * Scales, calculates absolute values, and converts the result to 8-bit.
     *
     * On each element of the input array, the function convertScaleAbs
     * performs three operations sequentially: scaling, taking an absolute
     * value, conversion to an unsigned 8-bit type:
     * \(\texttt{dst} (I)= \texttt{saturate\_cast&lt;uchar&gt;} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\)
     * In case of multi-channel arrays, the function processes each channel
     * independently. When the output is not 8-bit, the operation can be
     * emulated by calling the Mat::convertTo method (or by using matrix
     * expressions) and then by calculating an absolute value of the result.
     * For example:
     * <code>
     *     Mat_&lt;float&gt; A(30,30);
     *     randu(A, Scalar(-100), Scalar(100));
     *     Mat_&lt;float&gt; B = A*5 + 3;
     *     B = abs(B);
     *     // Mat_&lt;float&gt; B = abs(A*5+3) will also do the job,
     *     // but it will allocate a temporary matrix
     * </code>
     * @param src input array.
     * @param dst output array.
     * SEE:  Mat::convertTo, cv::abs(const Mat&amp;)
     */
    public static void convertScaleAbs(Mat src, Mat dst) {
        convertScaleAbs_2(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::copyMakeBorder(Mat src, Mat& dst, int top, int bottom, int left, int right, int borderType, Scalar value = Scalar())
    //

    /**
     * Forms a border around an image.
     *
     * The function copies the source image into the middle of the destination image. The areas to the
     * left, to the right, above and below the copied source image will be filled with extrapolated
     * pixels. This is not what filtering functions based on it do (they extrapolate pixels on-fly), but
     * what other more complex functions, including your own, may do to simplify image boundary handling.
     *
     * The function supports the mode when src is already in the middle of dst . In this case, the
     * function does not copy src itself but simply constructs the border, for example:
     *
     * <code>
     *     // let border be the same in all directions
     *     int border=2;
     *     // constructs a larger image to fit both the image and the border
     *     Mat gray_buf(rgb.rows + border*2, rgb.cols + border*2, rgb.depth());
     *     // select the middle part of it w/o copying data
     *     Mat gray(gray_canvas, Rect(border, border, rgb.cols, rgb.rows));
     *     // convert image from RGB to grayscale
     *     cvtColor(rgb, gray, COLOR_RGB2GRAY);
     *     // form a border in-place
     *     copyMakeBorder(gray, gray_buf, border, border,
     *                    border, border, BORDER_REPLICATE);
     *     // now do some custom filtering ...
     *     ...
     * </code>
     * <b>Note:</b> When the source image is a part (ROI) of a bigger image, the function will try to use the
     * pixels outside of the ROI to form a border. To disable this feature and always do extrapolation, as
     * if src was not a ROI, use borderType | #BORDER_ISOLATED.
     *
     * @param src Source image.
     * @param dst Destination image of the same type as src and the size Size(src.cols+left+right,
     * src.rows+top+bottom) .
     * @param top the top pixels
     * @param bottom the bottom pixels
     * @param left the left pixels
     * @param right Parameter specifying how many pixels in each direction from the source image rectangle
     * to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs
     * to be built.
     * @param borderType Border type. See borderInterpolate for details.
     * @param value Border value if borderType==BORDER_CONSTANT .
     *
     * SEE:  borderInterpolate
     */
    public static void copyMakeBorder(Mat src, Mat dst, int top, int bottom, int left, int right, int borderType, Scalar value) {
        copyMakeBorder_0(src.nativeObj, dst.nativeObj, top, bottom, left, right, borderType, value.val[0], value.val[1], value.val[2], value.val[3]);
    }

    /**
     * Forms a border around an image.
     *
     * The function copies the source image into the middle of the destination image. The areas to the
     * left, to the right, above and below the copied source image will be filled with extrapolated
     * pixels. This is not what filtering functions based on it do (they extrapolate pixels on-fly), but
     * what other more complex functions, including your own, may do to simplify image boundary handling.
     *
     * The function supports the mode when src is already in the middle of dst . In this case, the
     * function does not copy src itself but simply constructs the border, for example:
     *
     * <code>
     *     // let border be the same in all directions
     *     int border=2;
     *     // constructs a larger image to fit both the image and the border
     *     Mat gray_buf(rgb.rows + border*2, rgb.cols + border*2, rgb.depth());
     *     // select the middle part of it w/o copying data
     *     Mat gray(gray_canvas, Rect(border, border, rgb.cols, rgb.rows));
     *     // convert image from RGB to grayscale
     *     cvtColor(rgb, gray, COLOR_RGB2GRAY);
     *     // form a border in-place
     *     copyMakeBorder(gray, gray_buf, border, border,
     *                    border, border, BORDER_REPLICATE);
     *     // now do some custom filtering ...
     *     ...
     * </code>
     * <b>Note:</b> When the source image is a part (ROI) of a bigger image, the function will try to use the
     * pixels outside of the ROI to form a border. To disable this feature and always do extrapolation, as
     * if src was not a ROI, use borderType | #BORDER_ISOLATED.
     *
     * @param src Source image.
     * @param dst Destination image of the same type as src and the size Size(src.cols+left+right,
     * src.rows+top+bottom) .
     * @param top the top pixels
     * @param bottom the bottom pixels
     * @param left the left pixels
     * @param right Parameter specifying how many pixels in each direction from the source image rectangle
     * to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs
     * to be built.
     * @param borderType Border type. See borderInterpolate for details.
     *
     * SEE:  borderInterpolate
     */
    public static void copyMakeBorder(Mat src, Mat dst, int top, int bottom, int left, int right, int borderType) {
        copyMakeBorder_1(src.nativeObj, dst.nativeObj, top, bottom, left, right, borderType);
    }


    //
    // C++:  void cv::copyTo(Mat src, Mat& dst, Mat mask)
    //

    /**
     *  This is an overloaded member function, provided for convenience (python)
     * Copies the matrix to another one.
     * When the operation mask is specified, if the Mat::create call shown above reallocates the matrix, the newly allocated matrix is initialized with all zeros before copying the data.
     * @param src source matrix.
     * @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
     * reallocated.
     * @param mask Operation mask of the same size as \*this. Its non-zero elements indicate which matrix
     * elements need to be copied. The mask has to be of type CV_8U and can have 1 or multiple channels.
     */
    public static void copyTo(Mat src, Mat dst, Mat mask) {
        copyTo_0(src.nativeObj, dst.nativeObj, mask.nativeObj);
    }


    //
    // C++:  void cv::dct(Mat src, Mat& dst, int flags = 0)
    //

    /**
     * Performs a forward or inverse discrete Cosine transform of 1D or 2D array.
     *
     * The function cv::dct performs a forward or inverse discrete Cosine transform (DCT) of a 1D or 2D
     * floating-point array:
     * <ul>
     *   <li>
     *    Forward Cosine transform of a 1D vector of N elements:
     *     \(Y = C^{(N)}  \cdot X\)
     *     where
     *     \(C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right )\)
     *     and
     *     \(\alpha_0=1\), \(\alpha_j=2\) for *j &gt; 0*.
     *   </li>
     *   <li>
     *    Inverse Cosine transform of a 1D vector of N elements:
     *     \(X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y\)
     *     (since \(C^{(N)}\) is an orthogonal matrix, \(C^{(N)} \cdot \left(C^{(N)}\right)^T = I\) )
     *   </li>
     *   <li>
     *    Forward 2D Cosine transform of M x N matrix:
     *     \(Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T\)
     *   </li>
     *   <li>
     *    Inverse 2D Cosine transform of M x N matrix:
     *     \(X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)}\)
     *   </li>
     * </ul>
     *
     * The function chooses the mode of operation by looking at the flags and size of the input array:
     * <ul>
     *   <li>
     *    If (flags &amp; #DCT_INVERSE) == 0 , the function does a forward 1D or 2D transform. Otherwise, it
     *     is an inverse 1D or 2D transform.
     *   </li>
     *   <li>
     *    If (flags &amp; #DCT_ROWS) != 0 , the function performs a 1D transform of each row.
     *   </li>
     *   <li>
     *    If the array is a single column or a single row, the function performs a 1D transform.
     *   </li>
     *   <li>
     *    If none of the above is true, the function performs a 2D transform.
     *   </li>
     * </ul>
     *
     * <b>Note:</b> Currently dct supports even-size arrays (2, 4, 6 ...). For data analysis and approximation, you
     * can pad the array when necessary.
     * Also, the function performance depends very much, and not monotonically, on the array size (see
     * getOptimalDFTSize ). In the current implementation DCT of a vector of size N is calculated via DFT
     * of a vector of size N/2 . Thus, the optimal DCT size N1 &gt;= N can be calculated as:
     * <code>
     *     size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }
     *     N1 = getOptimalDCTSize(N);
     * </code>
     * @param src input floating-point array.
     * @param dst output array of the same size and type as src .
     * @param flags transformation flags as a combination of cv::DftFlags (DCT_*)
     * SEE: dft , getOptimalDFTSize , idct
     */
    public static void dct(Mat src, Mat dst, int flags) {
        dct_0(src.nativeObj, dst.nativeObj, flags);
    }

    /**
     * Performs a forward or inverse discrete Cosine transform of 1D or 2D array.
     *
     * The function cv::dct performs a forward or inverse discrete Cosine transform (DCT) of a 1D or 2D
     * floating-point array:
     * <ul>
     *   <li>
     *    Forward Cosine transform of a 1D vector of N elements:
     *     \(Y = C^{(N)}  \cdot X\)
     *     where
     *     \(C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right )\)
     *     and
     *     \(\alpha_0=1\), \(\alpha_j=2\) for *j &gt; 0*.
     *   </li>
     *   <li>
     *    Inverse Cosine transform of a 1D vector of N elements:
     *     \(X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y\)
     *     (since \(C^{(N)}\) is an orthogonal matrix, \(C^{(N)} \cdot \left(C^{(N)}\right)^T = I\) )
     *   </li>
     *   <li>
     *    Forward 2D Cosine transform of M x N matrix:
     *     \(Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T\)
     *   </li>
     *   <li>
     *    Inverse 2D Cosine transform of M x N matrix:
     *     \(X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)}\)
     *   </li>
     * </ul>
     *
     * The function chooses the mode of operation by looking at the flags and size of the input array:
     * <ul>
     *   <li>
     *    If (flags &amp; #DCT_INVERSE) == 0 , the function does a forward 1D or 2D transform. Otherwise, it
     *     is an inverse 1D or 2D transform.
     *   </li>
     *   <li>
     *    If (flags &amp; #DCT_ROWS) != 0 , the function performs a 1D transform of each row.
     *   </li>
     *   <li>
     *    If the array is a single column or a single row, the function performs a 1D transform.
     *   </li>
     *   <li>
     *    If none of the above is true, the function performs a 2D transform.
     *   </li>
     * </ul>
     *
     * <b>Note:</b> Currently dct supports even-size arrays (2, 4, 6 ...). For data analysis and approximation, you
     * can pad the array when necessary.
     * Also, the function performance depends very much, and not monotonically, on the array size (see
     * getOptimalDFTSize ). In the current implementation DCT of a vector of size N is calculated via DFT
     * of a vector of size N/2 . Thus, the optimal DCT size N1 &gt;= N can be calculated as:
     * <code>
     *     size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }
     *     N1 = getOptimalDCTSize(N);
     * </code>
     * @param src input floating-point array.
     * @param dst output array of the same size and type as src .
     * SEE: dft , getOptimalDFTSize , idct
     */
    public static void dct(Mat src, Mat dst) {
        dct_1(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::dft(Mat src, Mat& dst, int flags = 0, int nonzeroRows = 0)
    //

    /**
     * Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.
     *
     * The function cv::dft performs one of the following:
     * <ul>
     *   <li>
     *    Forward the Fourier transform of a 1D vector of N elements:
     *     \(Y = F^{(N)}  \cdot X,\)
     *     where \(F^{(N)}_{jk}=\exp(-2\pi i j k/N)\) and \(i=\sqrt{-1}\)
     *   </li>
     *   <li>
     *    Inverse the Fourier transform of a 1D vector of N elements:
     *     \(\begin{array}{l} X'=  \left (F^{(N)} \right )^{-1}  \cdot Y =  \left (F^{(N)} \right )^*  \cdot y  \\ X = (1/N)  \cdot X, \end{array}\)
     *     where \(F^*=\left(\textrm{Re}(F^{(N)})-\textrm{Im}(F^{(N)})\right)^T\)
     *   </li>
     *   <li>
     *    Forward the 2D Fourier transform of a M x N matrix:
     *     \(Y = F^{(M)}  \cdot X  \cdot F^{(N)}\)
     *   </li>
     *   <li>
     *    Inverse the 2D Fourier transform of a M x N matrix:
     *     \(\begin{array}{l} X'=  \left (F^{(M)} \right )^*  \cdot Y  \cdot \left (F^{(N)} \right )^* \\ X =  \frac{1}{M \cdot N} \cdot X' \end{array}\)
     *   </li>
     * </ul>
     *
     * In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input
     * spectrum of the inverse Fourier transform can be represented in a packed format called *CCS*
     * (complex-conjugate-symmetrical). It was borrowed from IPL (Intel\* Image Processing Library). Here
     * is how 2D *CCS* spectrum looks:
     * \(\begin{bmatrix} Re Y_{0,0} &amp; Re Y_{0,1} &amp; Im Y_{0,1} &amp; Re Y_{0,2} &amp; Im Y_{0,2} &amp;  \cdots &amp; Re Y_{0,N/2-1} &amp; Im Y_{0,N/2-1} &amp; Re Y_{0,N/2}  \\ Re Y_{1,0} &amp; Re Y_{1,1} &amp; Im Y_{1,1} &amp; Re Y_{1,2} &amp; Im Y_{1,2} &amp;  \cdots &amp; Re Y_{1,N/2-1} &amp; Im Y_{1,N/2-1} &amp; Re Y_{1,N/2}  \\ Im Y_{1,0} &amp; Re Y_{2,1} &amp; Im Y_{2,1} &amp; Re Y_{2,2} &amp; Im Y_{2,2} &amp;  \cdots &amp; Re Y_{2,N/2-1} &amp; Im Y_{2,N/2-1} &amp; Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &amp;  Re Y_{M-3,1}  &amp; Im Y_{M-3,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-3,N/2-1} &amp; Im Y_{M-3,N/2-1}&amp; Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &amp;  Re Y_{M-2,1}  &amp; Im Y_{M-2,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-2,N/2-1} &amp; Im Y_{M-2,N/2-1}&amp; Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &amp;  Re Y_{M-1,1} &amp;  Im Y_{M-1,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-1,N/2-1} &amp; Im Y_{M-1,N/2-1}&amp; Re Y_{M/2,N/2} \end{bmatrix}\)
     *
     * In case of 1D transform of a real vector, the output looks like the first row of the matrix above.
     *
     * So, the function chooses an operation mode depending on the flags and size of the input array:
     * <ul>
     *   <li>
     *    If #DFT_ROWS is set or the input array has a single row or single column, the function
     *     performs a 1D forward or inverse transform of each row of a matrix when #DFT_ROWS is set.
     *     Otherwise, it performs a 2D transform.
     *   </li>
     *   <li>
     *    If the input array is real and #DFT_INVERSE is not set, the function performs a forward 1D or
     *     2D transform:
     *   <ul>
     *     <li>
     *        When #DFT_COMPLEX_OUTPUT is set, the output is a complex matrix of the same size as
     *         input.
     *     </li>
     *     <li>
     *        When #DFT_COMPLEX_OUTPUT is not set, the output is a real matrix of the same size as
     *         input. In case of 2D transform, it uses the packed format as shown above. In case of a
     *         single 1D transform, it looks like the first row of the matrix above. In case of
     *         multiple 1D transforms (when using the #DFT_ROWS flag), each row of the output matrix
     *         looks like the first row of the matrix above.
     *     </li>
     *   </ul>
     *   <li>
     *    If the input array is complex and either #DFT_INVERSE or #DFT_REAL_OUTPUT are not set, the
     *     output is a complex array of the same size as input. The function performs a forward or
     *     inverse 1D or 2D transform of the whole input array or each row of the input array
     *     independently, depending on the flags DFT_INVERSE and DFT_ROWS.
     *   </li>
     *   <li>
     *    When #DFT_INVERSE is set and the input array is real, or it is complex but #DFT_REAL_OUTPUT
     *     is set, the output is a real array of the same size as input. The function performs a 1D or 2D
     *     inverse transformation of the whole input array or each individual row, depending on the flags
     *     #DFT_INVERSE and #DFT_ROWS.
     *   </li>
     * </ul>
     *
     * If #DFT_SCALE is set, the scaling is done after the transformation.
     *
     * Unlike dct , the function supports arrays of arbitrary size. But only those arrays are processed
     * efficiently, whose sizes can be factorized in a product of small prime numbers (2, 3, and 5 in the
     * current implementation). Such an efficient DFT size can be calculated using the getOptimalDFTSize
     * method.
     *
     * The sample below illustrates how to calculate a DFT-based convolution of two 2D real arrays:
     * <code>
     *     void convolveDFT(InputArray A, InputArray B, OutputArray C)
     *     {
     *         // reallocate the output array if needed
     *         C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
     *         Size dftSize;
     *         // calculate the size of DFT transform
     *         dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
     *         dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
     *
     *         // allocate temporary buffers and initialize them with 0's
     *         Mat tempA(dftSize, A.type(), Scalar::all(0));
     *         Mat tempB(dftSize, B.type(), Scalar::all(0));
     *
     *         // copy A and B to the top-left corners of tempA and tempB, respectively
     *         Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
     *         A.copyTo(roiA);
     *         Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
     *         B.copyTo(roiB);
     *
     *         // now transform the padded A &amp; B in-place;
     *         // use "nonzeroRows" hint for faster processing
     *         dft(tempA, tempA, 0, A.rows);
     *         dft(tempB, tempB, 0, B.rows);
     *
     *         // multiply the spectrums;
     *         // the function handles packed spectrum representations well
     *         mulSpectrums(tempA, tempB, tempA);
     *
     *         // transform the product back from the frequency domain.
     *         // Even though all the result rows will be non-zero,
     *         // you need only the first C.rows of them, and thus you
     *         // pass nonzeroRows == C.rows
     *         dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
     *
     *         // now copy the result back to C.
     *         tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
     *
     *         // all the temporary buffers will be deallocated automatically
     *     }
     * </code>
     * To optimize this sample, consider the following approaches:
     * <ul>
     *   <li>
     *    Since nonzeroRows != 0 is passed to the forward transform calls and since A and B are copied to
     *     the top-left corners of tempA and tempB, respectively, it is not necessary to clear the whole
     *     tempA and tempB. It is only necessary to clear the tempA.cols - A.cols ( tempB.cols - B.cols)
     *     rightmost columns of the matrices.
     *   </li>
     *   <li>
     *    This DFT-based convolution does not have to be applied to the whole big arrays, especially if B
     *     is significantly smaller than A or vice versa. Instead, you can calculate convolution by parts.
     *     To do this, you need to split the output array C into multiple tiles. For each tile, estimate
     *     which parts of A and B are required to calculate convolution in this tile. If the tiles in C are
     *     too small, the speed will decrease a lot because of repeated work. In the ultimate case, when
     *     each tile in C is a single pixel, the algorithm becomes equivalent to the naive convolution
     *     algorithm. If the tiles are too big, the temporary arrays tempA and tempB become too big and
     *     there is also a slowdown because of bad cache locality. So, there is an optimal tile size
     *     somewhere in the middle.
     *   </li>
     *   <li>
     *    If different tiles in C can be calculated in parallel and, thus, the convolution is done by
     *     parts, the loop can be threaded.
     *   </li>
     * </ul>
     *
     * All of the above improvements have been implemented in #matchTemplate and #filter2D . Therefore, by
     * using them, you can get the performance even better than with the above theoretically optimal
     * implementation. Though, those two functions actually calculate cross-correlation, not convolution,
     * so you need to "flip" the second convolution operand B vertically and horizontally using flip .
     * <b>Note:</b>
     * <ul>
     *   <li>
     *    An example using the discrete fourier transform can be found at
     *     opencv_source_code/samples/cpp/dft.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the dft functionality to perform Wiener deconvolution can be found
     *     at opencv_source/samples/python/deconvolution.py
     *   </li>
     *   <li>
     *    (Python) An example rearranging the quadrants of a Fourier image can be found at
     *     opencv_source/samples/python/dft.py
     * @param src input array that could be real or complex.
     * @param dst output array whose size and type depends on the flags .
     * @param flags transformation flags, representing a combination of the #DftFlags
     * @param nonzeroRows when the parameter is not zero, the function assumes that only the first
     * nonzeroRows rows of the input array (#DFT_INVERSE is not set) or only the first nonzeroRows of the
     * output array (#DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the
     * rows more efficiently and save some time; this technique is very useful for calculating array
     * cross-correlation or convolution using DFT.
     * SEE: dct , getOptimalDFTSize , mulSpectrums, filter2D , matchTemplate , flip , cartToPolar ,
     * magnitude , phase
     *   </li>
     * </ul>
     */
    public static void dft(Mat src, Mat dst, int flags, int nonzeroRows) {
        dft_0(src.nativeObj, dst.nativeObj, flags, nonzeroRows);
    }

    /**
     * Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.
     *
     * The function cv::dft performs one of the following:
     * <ul>
     *   <li>
     *    Forward the Fourier transform of a 1D vector of N elements:
     *     \(Y = F^{(N)}  \cdot X,\)
     *     where \(F^{(N)}_{jk}=\exp(-2\pi i j k/N)\) and \(i=\sqrt{-1}\)
     *   </li>
     *   <li>
     *    Inverse the Fourier transform of a 1D vector of N elements:
     *     \(\begin{array}{l} X'=  \left (F^{(N)} \right )^{-1}  \cdot Y =  \left (F^{(N)} \right )^*  \cdot y  \\ X = (1/N)  \cdot X, \end{array}\)
     *     where \(F^*=\left(\textrm{Re}(F^{(N)})-\textrm{Im}(F^{(N)})\right)^T\)
     *   </li>
     *   <li>
     *    Forward the 2D Fourier transform of a M x N matrix:
     *     \(Y = F^{(M)}  \cdot X  \cdot F^{(N)}\)
     *   </li>
     *   <li>
     *    Inverse the 2D Fourier transform of a M x N matrix:
     *     \(\begin{array}{l} X'=  \left (F^{(M)} \right )^*  \cdot Y  \cdot \left (F^{(N)} \right )^* \\ X =  \frac{1}{M \cdot N} \cdot X' \end{array}\)
     *   </li>
     * </ul>
     *
     * In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input
     * spectrum of the inverse Fourier transform can be represented in a packed format called *CCS*
     * (complex-conjugate-symmetrical). It was borrowed from IPL (Intel\* Image Processing Library). Here
     * is how 2D *CCS* spectrum looks:
     * \(\begin{bmatrix} Re Y_{0,0} &amp; Re Y_{0,1} &amp; Im Y_{0,1} &amp; Re Y_{0,2} &amp; Im Y_{0,2} &amp;  \cdots &amp; Re Y_{0,N/2-1} &amp; Im Y_{0,N/2-1} &amp; Re Y_{0,N/2}  \\ Re Y_{1,0} &amp; Re Y_{1,1} &amp; Im Y_{1,1} &amp; Re Y_{1,2} &amp; Im Y_{1,2} &amp;  \cdots &amp; Re Y_{1,N/2-1} &amp; Im Y_{1,N/2-1} &amp; Re Y_{1,N/2}  \\ Im Y_{1,0} &amp; Re Y_{2,1} &amp; Im Y_{2,1} &amp; Re Y_{2,2} &amp; Im Y_{2,2} &amp;  \cdots &amp; Re Y_{2,N/2-1} &amp; Im Y_{2,N/2-1} &amp; Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &amp;  Re Y_{M-3,1}  &amp; Im Y_{M-3,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-3,N/2-1} &amp; Im Y_{M-3,N/2-1}&amp; Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &amp;  Re Y_{M-2,1}  &amp; Im Y_{M-2,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-2,N/2-1} &amp; Im Y_{M-2,N/2-1}&amp; Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &amp;  Re Y_{M-1,1} &amp;  Im Y_{M-1,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-1,N/2-1} &amp; Im Y_{M-1,N/2-1}&amp; Re Y_{M/2,N/2} \end{bmatrix}\)
     *
     * In case of 1D transform of a real vector, the output looks like the first row of the matrix above.
     *
     * So, the function chooses an operation mode depending on the flags and size of the input array:
     * <ul>
     *   <li>
     *    If #DFT_ROWS is set or the input array has a single row or single column, the function
     *     performs a 1D forward or inverse transform of each row of a matrix when #DFT_ROWS is set.
     *     Otherwise, it performs a 2D transform.
     *   </li>
     *   <li>
     *    If the input array is real and #DFT_INVERSE is not set, the function performs a forward 1D or
     *     2D transform:
     *   <ul>
     *     <li>
     *        When #DFT_COMPLEX_OUTPUT is set, the output is a complex matrix of the same size as
     *         input.
     *     </li>
     *     <li>
     *        When #DFT_COMPLEX_OUTPUT is not set, the output is a real matrix of the same size as
     *         input. In case of 2D transform, it uses the packed format as shown above. In case of a
     *         single 1D transform, it looks like the first row of the matrix above. In case of
     *         multiple 1D transforms (when using the #DFT_ROWS flag), each row of the output matrix
     *         looks like the first row of the matrix above.
     *     </li>
     *   </ul>
     *   <li>
     *    If the input array is complex and either #DFT_INVERSE or #DFT_REAL_OUTPUT are not set, the
     *     output is a complex array of the same size as input. The function performs a forward or
     *     inverse 1D or 2D transform of the whole input array or each row of the input array
     *     independently, depending on the flags DFT_INVERSE and DFT_ROWS.
     *   </li>
     *   <li>
     *    When #DFT_INVERSE is set and the input array is real, or it is complex but #DFT_REAL_OUTPUT
     *     is set, the output is a real array of the same size as input. The function performs a 1D or 2D
     *     inverse transformation of the whole input array or each individual row, depending on the flags
     *     #DFT_INVERSE and #DFT_ROWS.
     *   </li>
     * </ul>
     *
     * If #DFT_SCALE is set, the scaling is done after the transformation.
     *
     * Unlike dct , the function supports arrays of arbitrary size. But only those arrays are processed
     * efficiently, whose sizes can be factorized in a product of small prime numbers (2, 3, and 5 in the
     * current implementation). Such an efficient DFT size can be calculated using the getOptimalDFTSize
     * method.
     *
     * The sample below illustrates how to calculate a DFT-based convolution of two 2D real arrays:
     * <code>
     *     void convolveDFT(InputArray A, InputArray B, OutputArray C)
     *     {
     *         // reallocate the output array if needed
     *         C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
     *         Size dftSize;
     *         // calculate the size of DFT transform
     *         dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
     *         dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
     *
     *         // allocate temporary buffers and initialize them with 0's
     *         Mat tempA(dftSize, A.type(), Scalar::all(0));
     *         Mat tempB(dftSize, B.type(), Scalar::all(0));
     *
     *         // copy A and B to the top-left corners of tempA and tempB, respectively
     *         Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
     *         A.copyTo(roiA);
     *         Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
     *         B.copyTo(roiB);
     *
     *         // now transform the padded A &amp; B in-place;
     *         // use "nonzeroRows" hint for faster processing
     *         dft(tempA, tempA, 0, A.rows);
     *         dft(tempB, tempB, 0, B.rows);
     *
     *         // multiply the spectrums;
     *         // the function handles packed spectrum representations well
     *         mulSpectrums(tempA, tempB, tempA);
     *
     *         // transform the product back from the frequency domain.
     *         // Even though all the result rows will be non-zero,
     *         // you need only the first C.rows of them, and thus you
     *         // pass nonzeroRows == C.rows
     *         dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
     *
     *         // now copy the result back to C.
     *         tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
     *
     *         // all the temporary buffers will be deallocated automatically
     *     }
     * </code>
     * To optimize this sample, consider the following approaches:
     * <ul>
     *   <li>
     *    Since nonzeroRows != 0 is passed to the forward transform calls and since A and B are copied to
     *     the top-left corners of tempA and tempB, respectively, it is not necessary to clear the whole
     *     tempA and tempB. It is only necessary to clear the tempA.cols - A.cols ( tempB.cols - B.cols)
     *     rightmost columns of the matrices.
     *   </li>
     *   <li>
     *    This DFT-based convolution does not have to be applied to the whole big arrays, especially if B
     *     is significantly smaller than A or vice versa. Instead, you can calculate convolution by parts.
     *     To do this, you need to split the output array C into multiple tiles. For each tile, estimate
     *     which parts of A and B are required to calculate convolution in this tile. If the tiles in C are
     *     too small, the speed will decrease a lot because of repeated work. In the ultimate case, when
     *     each tile in C is a single pixel, the algorithm becomes equivalent to the naive convolution
     *     algorithm. If the tiles are too big, the temporary arrays tempA and tempB become too big and
     *     there is also a slowdown because of bad cache locality. So, there is an optimal tile size
     *     somewhere in the middle.
     *   </li>
     *   <li>
     *    If different tiles in C can be calculated in parallel and, thus, the convolution is done by
     *     parts, the loop can be threaded.
     *   </li>
     * </ul>
     *
     * All of the above improvements have been implemented in #matchTemplate and #filter2D . Therefore, by
     * using them, you can get the performance even better than with the above theoretically optimal
     * implementation. Though, those two functions actually calculate cross-correlation, not convolution,
     * so you need to "flip" the second convolution operand B vertically and horizontally using flip .
     * <b>Note:</b>
     * <ul>
     *   <li>
     *    An example using the discrete fourier transform can be found at
     *     opencv_source_code/samples/cpp/dft.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the dft functionality to perform Wiener deconvolution can be found
     *     at opencv_source/samples/python/deconvolution.py
     *   </li>
     *   <li>
     *    (Python) An example rearranging the quadrants of a Fourier image can be found at
     *     opencv_source/samples/python/dft.py
     * @param src input array that could be real or complex.
     * @param dst output array whose size and type depends on the flags .
     * @param flags transformation flags, representing a combination of the #DftFlags
     * nonzeroRows rows of the input array (#DFT_INVERSE is not set) or only the first nonzeroRows of the
     * output array (#DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the
     * rows more efficiently and save some time; this technique is very useful for calculating array
     * cross-correlation or convolution using DFT.
     * SEE: dct , getOptimalDFTSize , mulSpectrums, filter2D , matchTemplate , flip , cartToPolar ,
     * magnitude , phase
     *   </li>
     * </ul>
     */
    public static void dft(Mat src, Mat dst, int flags) {
        dft_1(src.nativeObj, dst.nativeObj, flags);
    }

    /**
     * Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.
     *
     * The function cv::dft performs one of the following:
     * <ul>
     *   <li>
     *    Forward the Fourier transform of a 1D vector of N elements:
     *     \(Y = F^{(N)}  \cdot X,\)
     *     where \(F^{(N)}_{jk}=\exp(-2\pi i j k/N)\) and \(i=\sqrt{-1}\)
     *   </li>
     *   <li>
     *    Inverse the Fourier transform of a 1D vector of N elements:
     *     \(\begin{array}{l} X'=  \left (F^{(N)} \right )^{-1}  \cdot Y =  \left (F^{(N)} \right )^*  \cdot y  \\ X = (1/N)  \cdot X, \end{array}\)
     *     where \(F^*=\left(\textrm{Re}(F^{(N)})-\textrm{Im}(F^{(N)})\right)^T\)
     *   </li>
     *   <li>
     *    Forward the 2D Fourier transform of a M x N matrix:
     *     \(Y = F^{(M)}  \cdot X  \cdot F^{(N)}\)
     *   </li>
     *   <li>
     *    Inverse the 2D Fourier transform of a M x N matrix:
     *     \(\begin{array}{l} X'=  \left (F^{(M)} \right )^*  \cdot Y  \cdot \left (F^{(N)} \right )^* \\ X =  \frac{1}{M \cdot N} \cdot X' \end{array}\)
     *   </li>
     * </ul>
     *
     * In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input
     * spectrum of the inverse Fourier transform can be represented in a packed format called *CCS*
     * (complex-conjugate-symmetrical). It was borrowed from IPL (Intel\* Image Processing Library). Here
     * is how 2D *CCS* spectrum looks:
     * \(\begin{bmatrix} Re Y_{0,0} &amp; Re Y_{0,1} &amp; Im Y_{0,1} &amp; Re Y_{0,2} &amp; Im Y_{0,2} &amp;  \cdots &amp; Re Y_{0,N/2-1} &amp; Im Y_{0,N/2-1} &amp; Re Y_{0,N/2}  \\ Re Y_{1,0} &amp; Re Y_{1,1} &amp; Im Y_{1,1} &amp; Re Y_{1,2} &amp; Im Y_{1,2} &amp;  \cdots &amp; Re Y_{1,N/2-1} &amp; Im Y_{1,N/2-1} &amp; Re Y_{1,N/2}  \\ Im Y_{1,0} &amp; Re Y_{2,1} &amp; Im Y_{2,1} &amp; Re Y_{2,2} &amp; Im Y_{2,2} &amp;  \cdots &amp; Re Y_{2,N/2-1} &amp; Im Y_{2,N/2-1} &amp; Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &amp;  Re Y_{M-3,1}  &amp; Im Y_{M-3,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-3,N/2-1} &amp; Im Y_{M-3,N/2-1}&amp; Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &amp;  Re Y_{M-2,1}  &amp; Im Y_{M-2,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-2,N/2-1} &amp; Im Y_{M-2,N/2-1}&amp; Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &amp;  Re Y_{M-1,1} &amp;  Im Y_{M-1,1} &amp;  \hdotsfor{3} &amp; Re Y_{M-1,N/2-1} &amp; Im Y_{M-1,N/2-1}&amp; Re Y_{M/2,N/2} \end{bmatrix}\)
     *
     * In case of 1D transform of a real vector, the output looks like the first row of the matrix above.
     *
     * So, the function chooses an operation mode depending on the flags and size of the input array:
     * <ul>
     *   <li>
     *    If #DFT_ROWS is set or the input array has a single row or single column, the function
     *     performs a 1D forward or inverse transform of each row of a matrix when #DFT_ROWS is set.
     *     Otherwise, it performs a 2D transform.
     *   </li>
     *   <li>
     *    If the input array is real and #DFT_INVERSE is not set, the function performs a forward 1D or
     *     2D transform:
     *   <ul>
     *     <li>
     *        When #DFT_COMPLEX_OUTPUT is set, the output is a complex matrix of the same size as
     *         input.
     *     </li>
     *     <li>
     *        When #DFT_COMPLEX_OUTPUT is not set, the output is a real matrix of the same size as
     *         input. In case of 2D transform, it uses the packed format as shown above. In case of a
     *         single 1D transform, it looks like the first row of the matrix above. In case of
     *         multiple 1D transforms (when using the #DFT_ROWS flag), each row of the output matrix
     *         looks like the first row of the matrix above.
     *     </li>
     *   </ul>
     *   <li>
     *    If the input array is complex and either #DFT_INVERSE or #DFT_REAL_OUTPUT are not set, the
     *     output is a complex array of the same size as input. The function performs a forward or
     *     inverse 1D or 2D transform of the whole input array or each row of the input array
     *     independently, depending on the flags DFT_INVERSE and DFT_ROWS.
     *   </li>
     *   <li>
     *    When #DFT_INVERSE is set and the input array is real, or it is complex but #DFT_REAL_OUTPUT
     *     is set, the output is a real array of the same size as input. The function performs a 1D or 2D
     *     inverse transformation of the whole input array or each individual row, depending on the flags
     *     #DFT_INVERSE and #DFT_ROWS.
     *   </li>
     * </ul>
     *
     * If #DFT_SCALE is set, the scaling is done after the transformation.
     *
     * Unlike dct , the function supports arrays of arbitrary size. But only those arrays are processed
     * efficiently, whose sizes can be factorized in a product of small prime numbers (2, 3, and 5 in the
     * current implementation). Such an efficient DFT size can be calculated using the getOptimalDFTSize
     * method.
     *
     * The sample below illustrates how to calculate a DFT-based convolution of two 2D real arrays:
     * <code>
     *     void convolveDFT(InputArray A, InputArray B, OutputArray C)
     *     {
     *         // reallocate the output array if needed
     *         C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
     *         Size dftSize;
     *         // calculate the size of DFT transform
     *         dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
     *         dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
     *
     *         // allocate temporary buffers and initialize them with 0's
     *         Mat tempA(dftSize, A.type(), Scalar::all(0));
     *         Mat tempB(dftSize, B.type(), Scalar::all(0));
     *
     *         // copy A and B to the top-left corners of tempA and tempB, respectively
     *         Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
     *         A.copyTo(roiA);
     *         Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
     *         B.copyTo(roiB);
     *
     *         // now transform the padded A &amp; B in-place;
     *         // use "nonzeroRows" hint for faster processing
     *         dft(tempA, tempA, 0, A.rows);
     *         dft(tempB, tempB, 0, B.rows);
     *
     *         // multiply the spectrums;
     *         // the function handles packed spectrum representations well
     *         mulSpectrums(tempA, tempB, tempA);
     *
     *         // transform the product back from the frequency domain.
     *         // Even though all the result rows will be non-zero,
     *         // you need only the first C.rows of them, and thus you
     *         // pass nonzeroRows == C.rows
     *         dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
     *
     *         // now copy the result back to C.
     *         tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
     *
     *         // all the temporary buffers will be deallocated automatically
     *     }
     * </code>
     * To optimize this sample, consider the following approaches:
     * <ul>
     *   <li>
     *    Since nonzeroRows != 0 is passed to the forward transform calls and since A and B are copied to
     *     the top-left corners of tempA and tempB, respectively, it is not necessary to clear the whole
     *     tempA and tempB. It is only necessary to clear the tempA.cols - A.cols ( tempB.cols - B.cols)
     *     rightmost columns of the matrices.
     *   </li>
     *   <li>
     *    This DFT-based convolution does not have to be applied to the whole big arrays, especially if B
     *     is significantly smaller than A or vice versa. Instead, you can calculate convolution by parts.
     *     To do this, you need to split the output array C into multiple tiles. For each tile, estimate
     *     which parts of A and B are required to calculate convolution in this tile. If the tiles in C are
     *     too small, the speed will decrease a lot because of repeated work. In the ultimate case, when
     *     each tile in C is a single pixel, the algorithm becomes equivalent to the naive convolution
     *     algorithm. If the tiles are too big, the temporary arrays tempA and tempB become too big and
     *     there is also a slowdown because of bad cache locality. So, there is an optimal tile size
     *     somewhere in the middle.
     *   </li>
     *   <li>
     *    If different tiles in C can be calculated in parallel and, thus, the convolution is done by
     *     parts, the loop can be threaded.
     *   </li>
     * </ul>
     *
     * All of the above improvements have been implemented in #matchTemplate and #filter2D . Therefore, by
     * using them, you can get the performance even better than with the above theoretically optimal
     * implementation. Though, those two functions actually calculate cross-correlation, not convolution,
     * so you need to "flip" the second convolution operand B vertically and horizontally using flip .
     * <b>Note:</b>
     * <ul>
     *   <li>
     *    An example using the discrete fourier transform can be found at
     *     opencv_source_code/samples/cpp/dft.cpp
     *   </li>
     *   <li>
     *    (Python) An example using the dft functionality to perform Wiener deconvolution can be found
     *     at opencv_source/samples/python/deconvolution.py
     *   </li>
     *   <li>
     *    (Python) An example rearranging the quadrants of a Fourier image can be found at
     *     opencv_source/samples/python/dft.py
     * @param src input array that could be real or complex.
     * @param dst output array whose size and type depends on the flags .
     * nonzeroRows rows of the input array (#DFT_INVERSE is not set) or only the first nonzeroRows of the
     * output array (#DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the
     * rows more efficiently and save some time; this technique is very useful for calculating array
     * cross-correlation or convolution using DFT.
     * SEE: dct , getOptimalDFTSize , mulSpectrums, filter2D , matchTemplate , flip , cartToPolar ,
     * magnitude , phase
     *   </li>
     * </ul>
     */
    public static void dft(Mat src, Mat dst) {
        dft_2(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::divide(Mat src1, Mat src2, Mat& dst, double scale = 1, int dtype = -1)
    //

    /**
     * Performs per-element division of two arrays or a scalar by an array.
     *
     * The function cv::divide divides one array by another:
     * \(\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\)
     * or a scalar by an array when there is no src1 :
     * \(\texttt{dst(I) = saturate(scale/src2(I))}\)
     *
     * Different channels of multi-channel arrays are processed independently.
     *
     * For integer types when src2(I) is zero, dst(I) will also be zero.
     *
     * <b>Note:</b> In case of floating point data there is no special defined behavior for zero src2(I) values.
     * Regular floating-point division is used.
     * Expect correct IEEE-754 behaviour for floating-point data (with NaN, Inf result values).
     *
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array.
     * @param src2 second input array of the same size and type as src1.
     * @param scale scalar factor.
     * @param dst output array of the same size and type as src2.
     * @param dtype optional depth of the output array; if -1, dst will have depth src2.depth(), but in
     * case of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth().
     * SEE:  multiply, add, subtract
     */
    public static void divide(Mat src1, Mat src2, Mat dst, double scale, int dtype) {
        divide_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, scale, dtype);
    }

    /**
     * Performs per-element division of two arrays or a scalar by an array.
     *
     * The function cv::divide divides one array by another:
     * \(\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\)
     * or a scalar by an array when there is no src1 :
     * \(\texttt{dst(I) = saturate(scale/src2(I))}\)
     *
     * Different channels of multi-channel arrays are processed independently.
     *
     * For integer types when src2(I) is zero, dst(I) will also be zero.
     *
     * <b>Note:</b> In case of floating point data there is no special defined behavior for zero src2(I) values.
     * Regular floating-point division is used.
     * Expect correct IEEE-754 behaviour for floating-point data (with NaN, Inf result values).
     *
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array.
     * @param src2 second input array of the same size and type as src1.
     * @param scale scalar factor.
     * @param dst output array of the same size and type as src2.
     * case of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth().
     * SEE:  multiply, add, subtract
     */
    public static void divide(Mat src1, Mat src2, Mat dst, double scale) {
        divide_1(src1.nativeObj, src2.nativeObj, dst.nativeObj, scale);
    }

    /**
     * Performs per-element division of two arrays or a scalar by an array.
     *
     * The function cv::divide divides one array by another:
     * \(\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\)
     * or a scalar by an array when there is no src1 :
     * \(\texttt{dst(I) = saturate(scale/src2(I))}\)
     *
     * Different channels of multi-channel arrays are processed independently.
     *
     * For integer types when src2(I) is zero, dst(I) will also be zero.
     *
     * <b>Note:</b> In case of floating point data there is no special defined behavior for zero src2(I) values.
     * Regular floating-point division is used.
     * Expect correct IEEE-754 behaviour for floating-point data (with NaN, Inf result values).
     *
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array.
     * @param src2 second input array of the same size and type as src1.
     * @param dst output array of the same size and type as src2.
     * case of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth().
     * SEE:  multiply, add, subtract
     */
    public static void divide(Mat src1, Mat src2, Mat dst) {
        divide_2(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::divide(Mat src1, Scalar src2, Mat& dst, double scale = 1, int dtype = -1)
    //

    public static void divide(Mat src1, Scalar src2, Mat dst, double scale, int dtype) {
        divide_3(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, scale, dtype);
    }

    public static void divide(Mat src1, Scalar src2, Mat dst, double scale) {
        divide_4(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, scale);
    }

    public static void divide(Mat src1, Scalar src2, Mat dst) {
        divide_5(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj);
    }


    //
    // C++:  void cv::divide(double scale, Mat src2, Mat& dst, int dtype = -1)
    //

    public static void divide(double scale, Mat src2, Mat dst, int dtype) {
        divide_6(scale, src2.nativeObj, dst.nativeObj, dtype);
    }

    public static void divide(double scale, Mat src2, Mat dst) {
        divide_7(scale, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::eigenNonSymmetric(Mat src, Mat& eigenvalues, Mat& eigenvectors)
    //

    /**
     * Calculates eigenvalues and eigenvectors of a non-symmetric matrix (real eigenvalues only).
     *
     * <b>Note:</b> Assumes real eigenvalues.
     *
     * The function calculates eigenvalues and eigenvectors (optional) of the square matrix src:
     * <code>
     *     src*eigenvectors.row(i).t() = eigenvalues.at&lt;srcType&gt;(i)*eigenvectors.row(i).t()
     * </code>
     *
     * @param src input matrix (CV_32FC1 or CV_64FC1 type).
     * @param eigenvalues output vector of eigenvalues (type is the same type as src).
     * @param eigenvectors output matrix of eigenvectors (type is the same type as src). The eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding eigenvalues.
     * SEE: eigen
     */
    public static void eigenNonSymmetric(Mat src, Mat eigenvalues, Mat eigenvectors) {
        eigenNonSymmetric_0(src.nativeObj, eigenvalues.nativeObj, eigenvectors.nativeObj);
    }


    //
    // C++:  void cv::exp(Mat src, Mat& dst)
    //

    /**
     * Calculates the exponent of every array element.
     *
     * The function cv::exp calculates the exponent of every element of the input
     * array:
     * \(\texttt{dst} [I] = e^{ src(I) }\)
     *
     * The maximum relative error is about 7e-6 for single-precision input and
     * less than 1e-10 for double-precision input. Currently, the function
     * converts denormalized values to zeros on output. Special values (NaN,
     * Inf) are not handled.
     * @param src input array.
     * @param dst output array of the same size and type as src.
     * SEE: log , cartToPolar , polarToCart , phase , pow , sqrt , magnitude
     */
    public static void exp(Mat src, Mat dst) {
        exp_0(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::extractChannel(Mat src, Mat& dst, int coi)
    //

    /**
     * Extracts a single channel from src (coi is 0-based index)
     * @param src input array
     * @param dst output array
     * @param coi index of channel to extract
     * SEE: mixChannels, split
     */
    public static void extractChannel(Mat src, Mat dst, int coi) {
        extractChannel_0(src.nativeObj, dst.nativeObj, coi);
    }


    //
    // C++:  void cv::findNonZero(Mat src, Mat& idx)
    //

    /**
     * Returns the list of locations of non-zero pixels
     *
     * Given a binary matrix (likely returned from an operation such
     * as threshold(), compare(), &gt;, ==, etc, return all of
     * the non-zero indices as a cv::Mat or std::vector&lt;cv::Point&gt; (x,y)
     * For example:
     * <code>
     *     cv::Mat binaryImage; // input, binary image
     *     cv::Mat locations;   // output, locations of non-zero pixels
     *     cv::findNonZero(binaryImage, locations);
     *
     *     // access pixel coordinates
     *     Point pnt = locations.at&lt;Point&gt;(i);
     * </code>
     * or
     * <code>
     *     cv::Mat binaryImage; // input, binary image
     *     vector&lt;Point&gt; locations;   // output, locations of non-zero pixels
     *     cv::findNonZero(binaryImage, locations);
     *
     *     // access pixel coordinates
     *     Point pnt = locations[i];
     * </code>
     * @param src single-channel array
     * @param idx the output array, type of cv::Mat or std::vector&lt;Point&gt;, corresponding to non-zero indices in the input
     */
    public static void findNonZero(Mat src, Mat idx) {
        findNonZero_0(src.nativeObj, idx.nativeObj);
    }


    //
    // C++:  void cv::flip(Mat src, Mat& dst, int flipCode)
    //

    /**
     * Flips a 2D array around vertical, horizontal, or both axes.
     *
     * The function cv::flip flips the array in one of three different ways (row
     * and column indices are 0-based):
     * \(\texttt{dst} _{ij} =
     * \left\{
     * \begin{array}{l l}
     * \texttt{src} _{\texttt{src.rows}-i-1,j} &amp; if\;  \texttt{flipCode} = 0 \\
     * \texttt{src} _{i, \texttt{src.cols} -j-1} &amp; if\;  \texttt{flipCode} &gt; 0 \\
     * \texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} &amp; if\; \texttt{flipCode} &lt; 0 \\
     * \end{array}
     * \right.\)
     * The example scenarios of using the function are the following:
     * Vertical flipping of the image (flipCode == 0) to switch between
     *     top-left and bottom-left image origin. This is a typical operation
     *     in video processing on Microsoft Windows\* OS.
     * Horizontal flipping of the image with the subsequent horizontal
     *     shift and absolute difference calculation to check for a
     *     vertical-axis symmetry (flipCode &gt; 0).
     * Simultaneous horizontal and vertical flipping of the image with
     *     the subsequent shift and absolute difference calculation to check
     *     for a central symmetry (flipCode &lt; 0).
     * Reversing the order of point arrays (flipCode &gt; 0 or
     *     flipCode == 0).
     * @param src input array.
     * @param dst output array of the same size and type as src.
     * @param flipCode a flag to specify how to flip the array; 0 means
     * flipping around the x-axis and positive value (for example, 1) means
     * flipping around y-axis. Negative value (for example, -1) means flipping
     * around both axes.
     * SEE: transpose , repeat , completeSymm
     */
    public static void flip(Mat src, Mat dst, int flipCode) {
        flip_0(src.nativeObj, dst.nativeObj, flipCode);
    }


    //
    // C++:  void cv::gemm(Mat src1, Mat src2, double alpha, Mat src3, double beta, Mat& dst, int flags = 0)
    //

    /**
     * Performs generalized matrix multiplication.
     *
     * The function cv::gemm performs generalized matrix multiplication similar to the
     * gemm functions in BLAS level 3. For example,
     * {@code gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T)}
     * corresponds to
     * \(\texttt{dst} =  \texttt{alpha} \cdot \texttt{src1} ^T  \cdot \texttt{src2} +  \texttt{beta} \cdot \texttt{src3} ^T\)
     *
     * In case of complex (two-channel) data, performed a complex matrix
     * multiplication.
     *
     * The function can be replaced with a matrix expression. For example, the
     * above call can be replaced with:
     * <code>
     *     dst = alpha*src1.t()*src2 + beta*src3.t();
     * </code>
     * @param src1 first multiplied input matrix that could be real(CV_32FC1,
     * CV_64FC1) or complex(CV_32FC2, CV_64FC2).
     * @param src2 second multiplied input matrix of the same type as src1.
     * @param alpha weight of the matrix product.
     * @param src3 third optional delta matrix added to the matrix product; it
     * should have the same type as src1 and src2.
     * @param beta weight of src3.
     * @param dst output matrix; it has the proper size and the same type as
     * input matrices.
     * @param flags operation flags (cv::GemmFlags)
     * SEE: mulTransposed , transform
     */
    public static void gemm(Mat src1, Mat src2, double alpha, Mat src3, double beta, Mat dst, int flags) {
        gemm_0(src1.nativeObj, src2.nativeObj, alpha, src3.nativeObj, beta, dst.nativeObj, flags);
    }

    /**
     * Performs generalized matrix multiplication.
     *
     * The function cv::gemm performs generalized matrix multiplication similar to the
     * gemm functions in BLAS level 3. For example,
     * {@code gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T)}
     * corresponds to
     * \(\texttt{dst} =  \texttt{alpha} \cdot \texttt{src1} ^T  \cdot \texttt{src2} +  \texttt{beta} \cdot \texttt{src3} ^T\)
     *
     * In case of complex (two-channel) data, performed a complex matrix
     * multiplication.
     *
     * The function can be replaced with a matrix expression. For example, the
     * above call can be replaced with:
     * <code>
     *     dst = alpha*src1.t()*src2 + beta*src3.t();
     * </code>
     * @param src1 first multiplied input matrix that could be real(CV_32FC1,
     * CV_64FC1) or complex(CV_32FC2, CV_64FC2).
     * @param src2 second multiplied input matrix of the same type as src1.
     * @param alpha weight of the matrix product.
     * @param src3 third optional delta matrix added to the matrix product; it
     * should have the same type as src1 and src2.
     * @param beta weight of src3.
     * @param dst output matrix; it has the proper size and the same type as
     * input matrices.
     * SEE: mulTransposed , transform
     */
    public static void gemm(Mat src1, Mat src2, double alpha, Mat src3, double beta, Mat dst) {
        gemm_1(src1.nativeObj, src2.nativeObj, alpha, src3.nativeObj, beta, dst.nativeObj);
    }


    //
    // C++:  void cv::hconcat(vector_Mat src, Mat& dst)
    //

    /**
     *
     *  <code>
     *     std::vector&lt;cv::Mat&gt; matrices = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),
     *                                       cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),
     *                                       cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};
     *
     *     cv::Mat out;
     *     cv::hconcat( matrices, out );
     *     //out:
     *     //[1, 2, 3;
     *     // 1, 2, 3;
     *     // 1, 2, 3;
     *     // 1, 2, 3]
     *  </code>
     *  @param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.
     *  @param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.
     * same depth.
     */
    public static void hconcat(List<Mat> src, Mat dst) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        hconcat_0(src_mat.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::idct(Mat src, Mat& dst, int flags = 0)
    //

    /**
     * Calculates the inverse Discrete Cosine Transform of a 1D or 2D array.
     *
     * idct(src, dst, flags) is equivalent to dct(src, dst, flags | DCT_INVERSE).
     * @param src input floating-point single-channel array.
     * @param dst output array of the same size and type as src.
     * @param flags operation flags.
     * SEE:  dct, dft, idft, getOptimalDFTSize
     */
    public static void idct(Mat src, Mat dst, int flags) {
        idct_0(src.nativeObj, dst.nativeObj, flags);
    }

    /**
     * Calculates the inverse Discrete Cosine Transform of a 1D or 2D array.
     *
     * idct(src, dst, flags) is equivalent to dct(src, dst, flags | DCT_INVERSE).
     * @param src input floating-point single-channel array.
     * @param dst output array of the same size and type as src.
     * SEE:  dct, dft, idft, getOptimalDFTSize
     */
    public static void idct(Mat src, Mat dst) {
        idct_1(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::idft(Mat src, Mat& dst, int flags = 0, int nonzeroRows = 0)
    //

    /**
     * Calculates the inverse Discrete Fourier Transform of a 1D or 2D array.
     *
     * idft(src, dst, flags) is equivalent to dft(src, dst, flags | #DFT_INVERSE) .
     * <b>Note:</b> None of dft and idft scales the result by default. So, you should pass #DFT_SCALE to one of
     * dft or idft explicitly to make these transforms mutually inverse.
     * SEE: dft, dct, idct, mulSpectrums, getOptimalDFTSize
     * @param src input floating-point real or complex array.
     * @param dst output array whose size and type depend on the flags.
     * @param flags operation flags (see dft and #DftFlags).
     * @param nonzeroRows number of dst rows to process; the rest of the rows have undefined content (see
     * the convolution sample in dft description.
     */
    public static void idft(Mat src, Mat dst, int flags, int nonzeroRows) {
        idft_0(src.nativeObj, dst.nativeObj, flags, nonzeroRows);
    }

    /**
     * Calculates the inverse Discrete Fourier Transform of a 1D or 2D array.
     *
     * idft(src, dst, flags) is equivalent to dft(src, dst, flags | #DFT_INVERSE) .
     * <b>Note:</b> None of dft and idft scales the result by default. So, you should pass #DFT_SCALE to one of
     * dft or idft explicitly to make these transforms mutually inverse.
     * SEE: dft, dct, idct, mulSpectrums, getOptimalDFTSize
     * @param src input floating-point real or complex array.
     * @param dst output array whose size and type depend on the flags.
     * @param flags operation flags (see dft and #DftFlags).
     * the convolution sample in dft description.
     */
    public static void idft(Mat src, Mat dst, int flags) {
        idft_1(src.nativeObj, dst.nativeObj, flags);
    }

    /**
     * Calculates the inverse Discrete Fourier Transform of a 1D or 2D array.
     *
     * idft(src, dst, flags) is equivalent to dft(src, dst, flags | #DFT_INVERSE) .
     * <b>Note:</b> None of dft and idft scales the result by default. So, you should pass #DFT_SCALE to one of
     * dft or idft explicitly to make these transforms mutually inverse.
     * SEE: dft, dct, idct, mulSpectrums, getOptimalDFTSize
     * @param src input floating-point real or complex array.
     * @param dst output array whose size and type depend on the flags.
     * the convolution sample in dft description.
     */
    public static void idft(Mat src, Mat dst) {
        idft_2(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::inRange(Mat src, Scalar lowerb, Scalar upperb, Mat& dst)
    //

    /**
     *  Checks if array elements lie between the elements of two other arrays.
     *
     * The function checks the range as follows:
     * <ul>
     *   <li>
     *    For every element of a single-channel input array:
     *     \(\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0\)
     *   </li>
     *   <li>
     *    For two-channel arrays:
     *     \(\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0  \land \texttt{lowerb} (I)_1  \leq \texttt{src} (I)_1 \leq  \texttt{upperb} (I)_1\)
     *   </li>
     *   <li>
     *    and so forth.
     *   </li>
     * </ul>
     *
     * That is, dst (I) is set to 255 (all 1 -bits) if src (I) is within the
     * specified 1D, 2D, 3D, ... box and 0 otherwise.
     *
     * When the lower and/or upper boundary parameters are scalars, the indexes
     * (I) at lowerb and upperb in the above formulas should be omitted.
     * @param src first input array.
     * @param lowerb inclusive lower boundary array or a scalar.
     * @param upperb inclusive upper boundary array or a scalar.
     * @param dst output array of the same size as src and CV_8U type.
     */
    public static void inRange(Mat src, Scalar lowerb, Scalar upperb, Mat dst) {
        inRange_0(src.nativeObj, lowerb.val[0], lowerb.val[1], lowerb.val[2], lowerb.val[3], upperb.val[0], upperb.val[1], upperb.val[2], upperb.val[3], dst.nativeObj);
    }


    //
    // C++:  void cv::insertChannel(Mat src, Mat& dst, int coi)
    //

    /**
     * Inserts a single channel to dst (coi is 0-based index)
     * @param src input array
     * @param dst output array
     * @param coi index of channel for insertion
     * SEE: mixChannels, merge
     */
    public static void insertChannel(Mat src, Mat dst, int coi) {
        insertChannel_0(src.nativeObj, dst.nativeObj, coi);
    }


    //
    // C++:  void cv::log(Mat src, Mat& dst)
    //

    /**
     * Calculates the natural logarithm of every array element.
     *
     * The function cv::log calculates the natural logarithm of every element of the input array:
     * \(\texttt{dst} (I) =  \log (\texttt{src}(I)) \)
     *
     * Output on zero, negative and special (NaN, Inf) values is undefined.
     *
     * @param src input array.
     * @param dst output array of the same size and type as src .
     * SEE: exp, cartToPolar, polarToCart, phase, pow, sqrt, magnitude
     */
    public static void log(Mat src, Mat dst) {
        log_0(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::magnitude(Mat x, Mat y, Mat& magnitude)
    //

    /**
     * Calculates the magnitude of 2D vectors.
     *
     * The function cv::magnitude calculates the magnitude of 2D vectors formed
     * from the corresponding elements of x and y arrays:
     * \(\texttt{dst} (I) =  \sqrt{\texttt{x}(I)^2 + \texttt{y}(I)^2}\)
     * @param x floating-point array of x-coordinates of the vectors.
     * @param y floating-point array of y-coordinates of the vectors; it must
     * have the same size as x.
     * @param magnitude output array of the same size and type as x.
     * SEE: cartToPolar, polarToCart, phase, sqrt
     */
    public static void magnitude(Mat x, Mat y, Mat magnitude) {
        magnitude_0(x.nativeObj, y.nativeObj, magnitude.nativeObj);
    }


    //
    // C++:  void cv::max(Mat src1, Mat src2, Mat& dst)
    //

    /**
     * Calculates per-element maximum of two arrays or an array and a scalar.
     *
     * The function cv::max calculates the per-element maximum of two arrays:
     * \(\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))\)
     * or array and a scalar:
     * \(\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )\)
     * @param src1 first input array.
     * @param src2 second input array of the same size and type as src1 .
     * @param dst output array of the same size and type as src1.
     * SEE:  min, compare, inRange, minMaxLoc, REF: MatrixExpressions
     */
    public static void max(Mat src1, Mat src2, Mat dst) {
        max_0(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::max(Mat src1, Scalar src2, Mat& dst)
    //

    public static void max(Mat src1, Scalar src2, Mat dst) {
        max_1(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj);
    }


    //
    // C++:  void cv::meanStdDev(Mat src, vector_double& mean, vector_double& stddev, Mat mask = Mat())
    //

    /**
     * Calculates a mean and standard deviation of array elements.
     *
     * The function cv::meanStdDev calculates the mean and the standard deviation M
     * of array elements independently for each channel and returns it via the
     * output parameters:
     * \(\begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2}{N}} \end{array}\)
     * When all the mask elements are 0's, the function returns
     * mean=stddev=Scalar::all(0).
     * <b>Note:</b> The calculated standard deviation is only the diagonal of the
     * complete normalized covariance matrix. If the full matrix is needed, you
     * can reshape the multi-channel array M x N to the single-channel array
     * M\*N x mtx.channels() (only possible when the matrix is continuous) and
     * then pass the matrix to calcCovarMatrix .
     * @param src input array that should have from 1 to 4 channels so that the results can be stored in
     * Scalar_ 's.
     * @param mean output parameter: calculated mean value.
     * @param stddev output parameter: calculated standard deviation.
     * @param mask optional operation mask.
     * SEE:  countNonZero, mean, norm, minMaxLoc, calcCovarMatrix
     */
    public static void meanStdDev(Mat src, MatOfDouble mean, MatOfDouble stddev, Mat mask) {
        Mat mean_mat = mean;
        Mat stddev_mat = stddev;
        meanStdDev_0(src.nativeObj, mean_mat.nativeObj, stddev_mat.nativeObj, mask.nativeObj);
    }

    /**
     * Calculates a mean and standard deviation of array elements.
     *
     * The function cv::meanStdDev calculates the mean and the standard deviation M
     * of array elements independently for each channel and returns it via the
     * output parameters:
     * \(\begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2}{N}} \end{array}\)
     * When all the mask elements are 0's, the function returns
     * mean=stddev=Scalar::all(0).
     * <b>Note:</b> The calculated standard deviation is only the diagonal of the
     * complete normalized covariance matrix. If the full matrix is needed, you
     * can reshape the multi-channel array M x N to the single-channel array
     * M\*N x mtx.channels() (only possible when the matrix is continuous) and
     * then pass the matrix to calcCovarMatrix .
     * @param src input array that should have from 1 to 4 channels so that the results can be stored in
     * Scalar_ 's.
     * @param mean output parameter: calculated mean value.
     * @param stddev output parameter: calculated standard deviation.
     * SEE:  countNonZero, mean, norm, minMaxLoc, calcCovarMatrix
     */
    public static void meanStdDev(Mat src, MatOfDouble mean, MatOfDouble stddev) {
        Mat mean_mat = mean;
        Mat stddev_mat = stddev;
        meanStdDev_1(src.nativeObj, mean_mat.nativeObj, stddev_mat.nativeObj);
    }


    //
    // C++:  void cv::merge(vector_Mat mv, Mat& dst)
    //

    /**
     *
     * @param mv input vector of matrices to be merged; all the matrices in mv must have the same
     * size and the same depth.
     * @param dst output array of the same size and the same depth as mv[0]; The number of channels will
     * be the total number of channels in the matrix array.
     */
    public static void merge(List<Mat> mv, Mat dst) {
        Mat mv_mat = Converters.vector_Mat_to_Mat(mv);
        merge_0(mv_mat.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::min(Mat src1, Mat src2, Mat& dst)
    //

    /**
     * Calculates per-element minimum of two arrays or an array and a scalar.
     *
     * The function cv::min calculates the per-element minimum of two arrays:
     * \(\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))\)
     * or array and a scalar:
     * \(\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )\)
     * @param src1 first input array.
     * @param src2 second input array of the same size and type as src1.
     * @param dst output array of the same size and type as src1.
     * SEE: max, compare, inRange, minMaxLoc
     */
    public static void min(Mat src1, Mat src2, Mat dst) {
        min_0(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::min(Mat src1, Scalar src2, Mat& dst)
    //

    public static void min(Mat src1, Scalar src2, Mat dst) {
        min_1(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj);
    }


    //
    // C++:  void cv::mixChannels(vector_Mat src, vector_Mat dst, vector_int fromTo)
    //

    /**
     *
     * @param src input array or vector of matrices; all of the matrices must have the same size and the
     * same depth.
     * @param dst output array or vector of matrices; all the matrices <b>must be allocated</b>; their size and
     * depth must be the same as in src[0].
     * @param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
     * a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
     * dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
     * src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
     * src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
     * channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
     * filled with zero .
     */
    public static void mixChannels(List<Mat> src, List<Mat> dst, MatOfInt fromTo) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        Mat dst_mat = Converters.vector_Mat_to_Mat(dst);
        Mat fromTo_mat = fromTo;
        mixChannels_0(src_mat.nativeObj, dst_mat.nativeObj, fromTo_mat.nativeObj);
    }


    //
    // C++:  void cv::mulSpectrums(Mat a, Mat b, Mat& c, int flags, bool conjB = false)
    //

    /**
     * Performs the per-element multiplication of two Fourier spectrums.
     *
     * The function cv::mulSpectrums performs the per-element multiplication of the two CCS-packed or complex
     * matrices that are results of a real or complex Fourier transform.
     *
     * The function, together with dft and idft , may be used to calculate convolution (pass conjB=false )
     * or correlation (pass conjB=true ) of two arrays rapidly. When the arrays are complex, they are
     * simply multiplied (per element) with an optional conjugation of the second-array elements. When the
     * arrays are real, they are assumed to be CCS-packed (see dft for details).
     * @param a first input array.
     * @param b second input array of the same size and type as src1 .
     * @param c output array of the same size and type as src1 .
     * @param flags operation flags; currently, the only supported flag is cv::DFT_ROWS, which indicates that
     * each row of src1 and src2 is an independent 1D Fourier spectrum. If you do not want to use this flag, then simply add a {@code 0} as value.
     * @param conjB optional flag that conjugates the second input array before the multiplication (true)
     * or not (false).
     */
    public static void mulSpectrums(Mat a, Mat b, Mat c, int flags, boolean conjB) {
        mulSpectrums_0(a.nativeObj, b.nativeObj, c.nativeObj, flags, conjB);
    }

    /**
     * Performs the per-element multiplication of two Fourier spectrums.
     *
     * The function cv::mulSpectrums performs the per-element multiplication of the two CCS-packed or complex
     * matrices that are results of a real or complex Fourier transform.
     *
     * The function, together with dft and idft , may be used to calculate convolution (pass conjB=false )
     * or correlation (pass conjB=true ) of two arrays rapidly. When the arrays are complex, they are
     * simply multiplied (per element) with an optional conjugation of the second-array elements. When the
     * arrays are real, they are assumed to be CCS-packed (see dft for details).
     * @param a first input array.
     * @param b second input array of the same size and type as src1 .
     * @param c output array of the same size and type as src1 .
     * @param flags operation flags; currently, the only supported flag is cv::DFT_ROWS, which indicates that
     * each row of src1 and src2 is an independent 1D Fourier spectrum. If you do not want to use this flag, then simply add a {@code 0} as value.
     * or not (false).
     */
    public static void mulSpectrums(Mat a, Mat b, Mat c, int flags) {
        mulSpectrums_1(a.nativeObj, b.nativeObj, c.nativeObj, flags);
    }


    //
    // C++:  void cv::mulTransposed(Mat src, Mat& dst, bool aTa, Mat delta = Mat(), double scale = 1, int dtype = -1)
    //

    /**
     * Calculates the product of a matrix and its transposition.
     *
     * The function cv::mulTransposed calculates the product of src and its
     * transposition:
     * \(\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )\)
     * if aTa=true , and
     * \(\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T\)
     * otherwise. The function is used to calculate the covariance matrix. With
     * zero delta, it can be used as a faster substitute for general matrix
     * product A\*B when B=A'
     * @param src input single-channel matrix. Note that unlike gemm, the
     * function can multiply not only floating-point matrices.
     * @param dst output square matrix.
     * @param aTa Flag specifying the multiplication ordering. See the
     * description below.
     * @param delta Optional delta matrix subtracted from src before the
     * multiplication. When the matrix is empty ( delta=noArray() ), it is
     * assumed to be zero, that is, nothing is subtracted. If it has the same
     * size as src , it is simply subtracted. Otherwise, it is "repeated" (see
     * repeat ) to cover the full src and then subtracted. Type of the delta
     * matrix, when it is not empty, must be the same as the type of created
     * output matrix. See the dtype parameter description below.
     * @param scale Optional scale factor for the matrix product.
     * @param dtype Optional type of the output matrix. When it is negative,
     * the output matrix will have the same type as src . Otherwise, it will be
     * type=CV_MAT_DEPTH(dtype) that should be either CV_32F or CV_64F .
     * SEE: calcCovarMatrix, gemm, repeat, reduce
     */
    public static void mulTransposed(Mat src, Mat dst, boolean aTa, Mat delta, double scale, int dtype) {
        mulTransposed_0(src.nativeObj, dst.nativeObj, aTa, delta.nativeObj, scale, dtype);
    }

    /**
     * Calculates the product of a matrix and its transposition.
     *
     * The function cv::mulTransposed calculates the product of src and its
     * transposition:
     * \(\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )\)
     * if aTa=true , and
     * \(\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T\)
     * otherwise. The function is used to calculate the covariance matrix. With
     * zero delta, it can be used as a faster substitute for general matrix
     * product A\*B when B=A'
     * @param src input single-channel matrix. Note that unlike gemm, the
     * function can multiply not only floating-point matrices.
     * @param dst output square matrix.
     * @param aTa Flag specifying the multiplication ordering. See the
     * description below.
     * @param delta Optional delta matrix subtracted from src before the
     * multiplication. When the matrix is empty ( delta=noArray() ), it is
     * assumed to be zero, that is, nothing is subtracted. If it has the same
     * size as src , it is simply subtracted. Otherwise, it is "repeated" (see
     * repeat ) to cover the full src and then subtracted. Type of the delta
     * matrix, when it is not empty, must be the same as the type of created
     * output matrix. See the dtype parameter description below.
     * @param scale Optional scale factor for the matrix product.
     * the output matrix will have the same type as src . Otherwise, it will be
     * type=CV_MAT_DEPTH(dtype) that should be either CV_32F or CV_64F .
     * SEE: calcCovarMatrix, gemm, repeat, reduce
     */
    public static void mulTransposed(Mat src, Mat dst, boolean aTa, Mat delta, double scale) {
        mulTransposed_1(src.nativeObj, dst.nativeObj, aTa, delta.nativeObj, scale);
    }

    /**
     * Calculates the product of a matrix and its transposition.
     *
     * The function cv::mulTransposed calculates the product of src and its
     * transposition:
     * \(\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )\)
     * if aTa=true , and
     * \(\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T\)
     * otherwise. The function is used to calculate the covariance matrix. With
     * zero delta, it can be used as a faster substitute for general matrix
     * product A\*B when B=A'
     * @param src input single-channel matrix. Note that unlike gemm, the
     * function can multiply not only floating-point matrices.
     * @param dst output square matrix.
     * @param aTa Flag specifying the multiplication ordering. See the
     * description below.
     * @param delta Optional delta matrix subtracted from src before the
     * multiplication. When the matrix is empty ( delta=noArray() ), it is
     * assumed to be zero, that is, nothing is subtracted. If it has the same
     * size as src , it is simply subtracted. Otherwise, it is "repeated" (see
     * repeat ) to cover the full src and then subtracted. Type of the delta
     * matrix, when it is not empty, must be the same as the type of created
     * output matrix. See the dtype parameter description below.
     * the output matrix will have the same type as src . Otherwise, it will be
     * type=CV_MAT_DEPTH(dtype) that should be either CV_32F or CV_64F .
     * SEE: calcCovarMatrix, gemm, repeat, reduce
     */
    public static void mulTransposed(Mat src, Mat dst, boolean aTa, Mat delta) {
        mulTransposed_2(src.nativeObj, dst.nativeObj, aTa, delta.nativeObj);
    }

    /**
     * Calculates the product of a matrix and its transposition.
     *
     * The function cv::mulTransposed calculates the product of src and its
     * transposition:
     * \(\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )\)
     * if aTa=true , and
     * \(\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T\)
     * otherwise. The function is used to calculate the covariance matrix. With
     * zero delta, it can be used as a faster substitute for general matrix
     * product A\*B when B=A'
     * @param src input single-channel matrix. Note that unlike gemm, the
     * function can multiply not only floating-point matrices.
     * @param dst output square matrix.
     * @param aTa Flag specifying the multiplication ordering. See the
     * description below.
     * multiplication. When the matrix is empty ( delta=noArray() ), it is
     * assumed to be zero, that is, nothing is subtracted. If it has the same
     * size as src , it is simply subtracted. Otherwise, it is "repeated" (see
     * repeat ) to cover the full src and then subtracted. Type of the delta
     * matrix, when it is not empty, must be the same as the type of created
     * output matrix. See the dtype parameter description below.
     * the output matrix will have the same type as src . Otherwise, it will be
     * type=CV_MAT_DEPTH(dtype) that should be either CV_32F or CV_64F .
     * SEE: calcCovarMatrix, gemm, repeat, reduce
     */
    public static void mulTransposed(Mat src, Mat dst, boolean aTa) {
        mulTransposed_3(src.nativeObj, dst.nativeObj, aTa);
    }


    //
    // C++:  void cv::multiply(Mat src1, Mat src2, Mat& dst, double scale = 1, int dtype = -1)
    //

    /**
     * Calculates the per-element scaled product of two arrays.
     *
     * The function multiply calculates the per-element product of two arrays:
     *
     * \(\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\)
     *
     * There is also a REF: MatrixExpressions -friendly variant of the first function. See Mat::mul .
     *
     * For a not-per-element matrix product, see gemm .
     *
     * <b>Note:</b> Saturation is not applied when the output array has the depth
     * CV_32S. You may even get result of an incorrect sign in the case of
     * overflow.
     * @param src1 first input array.
     * @param src2 second input array of the same size and the same type as src1.
     * @param dst output array of the same size and type as src1.
     * @param scale optional scale factor.
     * @param dtype optional depth of the output array
     * SEE: add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,
     * Mat::convertTo
     */
    public static void multiply(Mat src1, Mat src2, Mat dst, double scale, int dtype) {
        multiply_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, scale, dtype);
    }

    /**
     * Calculates the per-element scaled product of two arrays.
     *
     * The function multiply calculates the per-element product of two arrays:
     *
     * \(\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\)
     *
     * There is also a REF: MatrixExpressions -friendly variant of the first function. See Mat::mul .
     *
     * For a not-per-element matrix product, see gemm .
     *
     * <b>Note:</b> Saturation is not applied when the output array has the depth
     * CV_32S. You may even get result of an incorrect sign in the case of
     * overflow.
     * @param src1 first input array.
     * @param src2 second input array of the same size and the same type as src1.
     * @param dst output array of the same size and type as src1.
     * @param scale optional scale factor.
     * SEE: add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,
     * Mat::convertTo
     */
    public static void multiply(Mat src1, Mat src2, Mat dst, double scale) {
        multiply_1(src1.nativeObj, src2.nativeObj, dst.nativeObj, scale);
    }

    /**
     * Calculates the per-element scaled product of two arrays.
     *
     * The function multiply calculates the per-element product of two arrays:
     *
     * \(\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\)
     *
     * There is also a REF: MatrixExpressions -friendly variant of the first function. See Mat::mul .
     *
     * For a not-per-element matrix product, see gemm .
     *
     * <b>Note:</b> Saturation is not applied when the output array has the depth
     * CV_32S. You may even get result of an incorrect sign in the case of
     * overflow.
     * @param src1 first input array.
     * @param src2 second input array of the same size and the same type as src1.
     * @param dst output array of the same size and type as src1.
     * SEE: add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,
     * Mat::convertTo
     */
    public static void multiply(Mat src1, Mat src2, Mat dst) {
        multiply_2(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::multiply(Mat src1, Scalar src2, Mat& dst, double scale = 1, int dtype = -1)
    //

    public static void multiply(Mat src1, Scalar src2, Mat dst, double scale, int dtype) {
        multiply_3(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, scale, dtype);
    }

    public static void multiply(Mat src1, Scalar src2, Mat dst, double scale) {
        multiply_4(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, scale);
    }

    public static void multiply(Mat src1, Scalar src2, Mat dst) {
        multiply_5(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj);
    }


    //
    // C++:  void cv::normalize(Mat src, Mat& dst, double alpha = 1, double beta = 0, int norm_type = NORM_L2, int dtype = -1, Mat mask = Mat())
    //

    /**
     * Normalizes the norm or value range of an array.
     *
     * The function cv::normalize normalizes scale and shift the input array elements so that
     * \(\| \texttt{dst} \| _{L_p}= \texttt{alpha}\)
     * (where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
     * \(\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\)
     *
     * when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
     * normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
     * sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
     * min-max but modify the whole array, you can use norm and Mat::convertTo.
     *
     * In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
     * the range transformation for sparse matrices is not allowed since it can shift the zero level.
     *
     * Possible usage with some positive example data:
     * <code>
     *     vector&lt;double&gt; positiveData = { 2.0, 8.0, 10.0 };
     *     vector&lt;double&gt; normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;
     *
     *     // Norm to probability (total count)
     *     // sum(numbers) = 20.0
     *     // 2.0      0.1     (2.0/20.0)
     *     // 8.0      0.4     (8.0/20.0)
     *     // 10.0     0.5     (10.0/20.0)
     *     normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);
     *
     *     // Norm to unit vector: ||positiveData|| = 1.0
     *     // 2.0      0.15
     *     // 8.0      0.62
     *     // 10.0     0.77
     *     normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);
     *
     *     // Norm to max element
     *     // 2.0      0.2     (2.0/10.0)
     *     // 8.0      0.8     (8.0/10.0)
     *     // 10.0     1.0     (10.0/10.0)
     *     normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);
     *
     *     // Norm to range [0.0;1.0]
     *     // 2.0      0.0     (shift to left border)
     *     // 8.0      0.75    (6.0/8.0)
     *     // 10.0     1.0     (shift to right border)
     *     normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
     * </code>
     *
     * @param src input array.
     * @param dst output array of the same size as src .
     * @param alpha norm value to normalize to or the lower range boundary in case of the range
     * normalization.
     * @param beta upper range boundary in case of the range normalization; it is not used for the norm
     * normalization.
     * @param norm_type normalization type (see cv::NormTypes).
     * @param dtype when negative, the output array has the same type as src; otherwise, it has the same
     * number of channels as src and the depth =CV_MAT_DEPTH(dtype).
     * @param mask optional operation mask.
     * SEE: norm, Mat::convertTo, SparseMat::convertTo
     */
    public static void normalize(Mat src, Mat dst, double alpha, double beta, int norm_type, int dtype, Mat mask) {
        normalize_0(src.nativeObj, dst.nativeObj, alpha, beta, norm_type, dtype, mask.nativeObj);
    }

    /**
     * Normalizes the norm or value range of an array.
     *
     * The function cv::normalize normalizes scale and shift the input array elements so that
     * \(\| \texttt{dst} \| _{L_p}= \texttt{alpha}\)
     * (where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
     * \(\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\)
     *
     * when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
     * normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
     * sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
     * min-max but modify the whole array, you can use norm and Mat::convertTo.
     *
     * In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
     * the range transformation for sparse matrices is not allowed since it can shift the zero level.
     *
     * Possible usage with some positive example data:
     * <code>
     *     vector&lt;double&gt; positiveData = { 2.0, 8.0, 10.0 };
     *     vector&lt;double&gt; normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;
     *
     *     // Norm to probability (total count)
     *     // sum(numbers) = 20.0
     *     // 2.0      0.1     (2.0/20.0)
     *     // 8.0      0.4     (8.0/20.0)
     *     // 10.0     0.5     (10.0/20.0)
     *     normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);
     *
     *     // Norm to unit vector: ||positiveData|| = 1.0
     *     // 2.0      0.15
     *     // 8.0      0.62
     *     // 10.0     0.77
     *     normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);
     *
     *     // Norm to max element
     *     // 2.0      0.2     (2.0/10.0)
     *     // 8.0      0.8     (8.0/10.0)
     *     // 10.0     1.0     (10.0/10.0)
     *     normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);
     *
     *     // Norm to range [0.0;1.0]
     *     // 2.0      0.0     (shift to left border)
     *     // 8.0      0.75    (6.0/8.0)
     *     // 10.0     1.0     (shift to right border)
     *     normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
     * </code>
     *
     * @param src input array.
     * @param dst output array of the same size as src .
     * @param alpha norm value to normalize to or the lower range boundary in case of the range
     * normalization.
     * @param beta upper range boundary in case of the range normalization; it is not used for the norm
     * normalization.
     * @param norm_type normalization type (see cv::NormTypes).
     * @param dtype when negative, the output array has the same type as src; otherwise, it has the same
     * number of channels as src and the depth =CV_MAT_DEPTH(dtype).
     * SEE: norm, Mat::convertTo, SparseMat::convertTo
     */
    public static void normalize(Mat src, Mat dst, double alpha, double beta, int norm_type, int dtype) {
        normalize_1(src.nativeObj, dst.nativeObj, alpha, beta, norm_type, dtype);
    }

    /**
     * Normalizes the norm or value range of an array.
     *
     * The function cv::normalize normalizes scale and shift the input array elements so that
     * \(\| \texttt{dst} \| _{L_p}= \texttt{alpha}\)
     * (where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
     * \(\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\)
     *
     * when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
     * normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
     * sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
     * min-max but modify the whole array, you can use norm and Mat::convertTo.
     *
     * In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
     * the range transformation for sparse matrices is not allowed since it can shift the zero level.
     *
     * Possible usage with some positive example data:
     * <code>
     *     vector&lt;double&gt; positiveData = { 2.0, 8.0, 10.0 };
     *     vector&lt;double&gt; normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;
     *
     *     // Norm to probability (total count)
     *     // sum(numbers) = 20.0
     *     // 2.0      0.1     (2.0/20.0)
     *     // 8.0      0.4     (8.0/20.0)
     *     // 10.0     0.5     (10.0/20.0)
     *     normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);
     *
     *     // Norm to unit vector: ||positiveData|| = 1.0
     *     // 2.0      0.15
     *     // 8.0      0.62
     *     // 10.0     0.77
     *     normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);
     *
     *     // Norm to max element
     *     // 2.0      0.2     (2.0/10.0)
     *     // 8.0      0.8     (8.0/10.0)
     *     // 10.0     1.0     (10.0/10.0)
     *     normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);
     *
     *     // Norm to range [0.0;1.0]
     *     // 2.0      0.0     (shift to left border)
     *     // 8.0      0.75    (6.0/8.0)
     *     // 10.0     1.0     (shift to right border)
     *     normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
     * </code>
     *
     * @param src input array.
     * @param dst output array of the same size as src .
     * @param alpha norm value to normalize to or the lower range boundary in case of the range
     * normalization.
     * @param beta upper range boundary in case of the range normalization; it is not used for the norm
     * normalization.
     * @param norm_type normalization type (see cv::NormTypes).
     * number of channels as src and the depth =CV_MAT_DEPTH(dtype).
     * SEE: norm, Mat::convertTo, SparseMat::convertTo
     */
    public static void normalize(Mat src, Mat dst, double alpha, double beta, int norm_type) {
        normalize_2(src.nativeObj, dst.nativeObj, alpha, beta, norm_type);
    }

    /**
     * Normalizes the norm or value range of an array.
     *
     * The function cv::normalize normalizes scale and shift the input array elements so that
     * \(\| \texttt{dst} \| _{L_p}= \texttt{alpha}\)
     * (where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
     * \(\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\)
     *
     * when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
     * normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
     * sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
     * min-max but modify the whole array, you can use norm and Mat::convertTo.
     *
     * In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
     * the range transformation for sparse matrices is not allowed since it can shift the zero level.
     *
     * Possible usage with some positive example data:
     * <code>
     *     vector&lt;double&gt; positiveData = { 2.0, 8.0, 10.0 };
     *     vector&lt;double&gt; normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;
     *
     *     // Norm to probability (total count)
     *     // sum(numbers) = 20.0
     *     // 2.0      0.1     (2.0/20.0)
     *     // 8.0      0.4     (8.0/20.0)
     *     // 10.0     0.5     (10.0/20.0)
     *     normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);
     *
     *     // Norm to unit vector: ||positiveData|| = 1.0
     *     // 2.0      0.15
     *     // 8.0      0.62
     *     // 10.0     0.77
     *     normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);
     *
     *     // Norm to max element
     *     // 2.0      0.2     (2.0/10.0)
     *     // 8.0      0.8     (8.0/10.0)
     *     // 10.0     1.0     (10.0/10.0)
     *     normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);
     *
     *     // Norm to range [0.0;1.0]
     *     // 2.0      0.0     (shift to left border)
     *     // 8.0      0.75    (6.0/8.0)
     *     // 10.0     1.0     (shift to right border)
     *     normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
     * </code>
     *
     * @param src input array.
     * @param dst output array of the same size as src .
     * @param alpha norm value to normalize to or the lower range boundary in case of the range
     * normalization.
     * @param beta upper range boundary in case of the range normalization; it is not used for the norm
     * normalization.
     * number of channels as src and the depth =CV_MAT_DEPTH(dtype).
     * SEE: norm, Mat::convertTo, SparseMat::convertTo
     */
    public static void normalize(Mat src, Mat dst, double alpha, double beta) {
        normalize_3(src.nativeObj, dst.nativeObj, alpha, beta);
    }

    /**
     * Normalizes the norm or value range of an array.
     *
     * The function cv::normalize normalizes scale and shift the input array elements so that
     * \(\| \texttt{dst} \| _{L_p}= \texttt{alpha}\)
     * (where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
     * \(\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\)
     *
     * when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
     * normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
     * sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
     * min-max but modify the whole array, you can use norm and Mat::convertTo.
     *
     * In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
     * the range transformation for sparse matrices is not allowed since it can shift the zero level.
     *
     * Possible usage with some positive example data:
     * <code>
     *     vector&lt;double&gt; positiveData = { 2.0, 8.0, 10.0 };
     *     vector&lt;double&gt; normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;
     *
     *     // Norm to probability (total count)
     *     // sum(numbers) = 20.0
     *     // 2.0      0.1     (2.0/20.0)
     *     // 8.0      0.4     (8.0/20.0)
     *     // 10.0     0.5     (10.0/20.0)
     *     normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);
     *
     *     // Norm to unit vector: ||positiveData|| = 1.0
     *     // 2.0      0.15
     *     // 8.0      0.62
     *     // 10.0     0.77
     *     normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);
     *
     *     // Norm to max element
     *     // 2.0      0.2     (2.0/10.0)
     *     // 8.0      0.8     (8.0/10.0)
     *     // 10.0     1.0     (10.0/10.0)
     *     normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);
     *
     *     // Norm to range [0.0;1.0]
     *     // 2.0      0.0     (shift to left border)
     *     // 8.0      0.75    (6.0/8.0)
     *     // 10.0     1.0     (shift to right border)
     *     normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
     * </code>
     *
     * @param src input array.
     * @param dst output array of the same size as src .
     * @param alpha norm value to normalize to or the lower range boundary in case of the range
     * normalization.
     * normalization.
     * number of channels as src and the depth =CV_MAT_DEPTH(dtype).
     * SEE: norm, Mat::convertTo, SparseMat::convertTo
     */
    public static void normalize(Mat src, Mat dst, double alpha) {
        normalize_4(src.nativeObj, dst.nativeObj, alpha);
    }

    /**
     * Normalizes the norm or value range of an array.
     *
     * The function cv::normalize normalizes scale and shift the input array elements so that
     * \(\| \texttt{dst} \| _{L_p}= \texttt{alpha}\)
     * (where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
     * \(\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\)
     *
     * when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
     * normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
     * sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
     * min-max but modify the whole array, you can use norm and Mat::convertTo.
     *
     * In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
     * the range transformation for sparse matrices is not allowed since it can shift the zero level.
     *
     * Possible usage with some positive example data:
     * <code>
     *     vector&lt;double&gt; positiveData = { 2.0, 8.0, 10.0 };
     *     vector&lt;double&gt; normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;
     *
     *     // Norm to probability (total count)
     *     // sum(numbers) = 20.0
     *     // 2.0      0.1     (2.0/20.0)
     *     // 8.0      0.4     (8.0/20.0)
     *     // 10.0     0.5     (10.0/20.0)
     *     normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);
     *
     *     // Norm to unit vector: ||positiveData|| = 1.0
     *     // 2.0      0.15
     *     // 8.0      0.62
     *     // 10.0     0.77
     *     normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);
     *
     *     // Norm to max element
     *     // 2.0      0.2     (2.0/10.0)
     *     // 8.0      0.8     (8.0/10.0)
     *     // 10.0     1.0     (10.0/10.0)
     *     normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);
     *
     *     // Norm to range [0.0;1.0]
     *     // 2.0      0.0     (shift to left border)
     *     // 8.0      0.75    (6.0/8.0)
     *     // 10.0     1.0     (shift to right border)
     *     normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
     * </code>
     *
     * @param src input array.
     * @param dst output array of the same size as src .
     * normalization.
     * normalization.
     * number of channels as src and the depth =CV_MAT_DEPTH(dtype).
     * SEE: norm, Mat::convertTo, SparseMat::convertTo
     */
    public static void normalize(Mat src, Mat dst) {
        normalize_5(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::patchNaNs(Mat& a, double val = 0)
    //

    /**
     * converts NaN's to the given number
     * @param a automatically generated
     * @param val automatically generated
     */
    public static void patchNaNs(Mat a, double val) {
        patchNaNs_0(a.nativeObj, val);
    }

    /**
     * converts NaN's to the given number
     * @param a automatically generated
     */
    public static void patchNaNs(Mat a) {
        patchNaNs_1(a.nativeObj);
    }


    //
    // C++:  void cv::perspectiveTransform(Mat src, Mat& dst, Mat m)
    //

    /**
     * Performs the perspective matrix transformation of vectors.
     *
     * The function cv::perspectiveTransform transforms every element of src by
     * treating it as a 2D or 3D vector, in the following way:
     * \((x, y, z)  \rightarrow (x'/w, y'/w, z'/w)\)
     * where
     * \((x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x &amp; y &amp; z &amp; 1  \end{bmatrix}\)
     * and
     * \(w =  \fork{w'}{if \(w' \ne 0\)}{\infty}{otherwise}\)
     *
     * Here a 3D vector transformation is shown. In case of a 2D vector
     * transformation, the z component is omitted.
     *
     * <b>Note:</b> The function transforms a sparse set of 2D or 3D vectors. If you
     * want to transform an image using perspective transformation, use
     * warpPerspective . If you have an inverse problem, that is, you want to
     * compute the most probable perspective transformation out of several
     * pairs of corresponding points, you can use getPerspectiveTransform or
     * findHomography .
     * @param src input two-channel or three-channel floating-point array; each
     * element is a 2D/3D vector to be transformed.
     * @param dst output array of the same size and type as src.
     * @param m 3x3 or 4x4 floating-point transformation matrix.
     * SEE:  transform, warpPerspective, getPerspectiveTransform, findHomography
     */
    public static void perspectiveTransform(Mat src, Mat dst, Mat m) {
        perspectiveTransform_0(src.nativeObj, dst.nativeObj, m.nativeObj);
    }


    //
    // C++:  void cv::phase(Mat x, Mat y, Mat& angle, bool angleInDegrees = false)
    //

    /**
     * Calculates the rotation angle of 2D vectors.
     *
     * The function cv::phase calculates the rotation angle of each 2D vector that
     * is formed from the corresponding elements of x and y :
     * \(\texttt{angle} (I) =  \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))\)
     *
     * The angle estimation accuracy is about 0.3 degrees. When x(I)=y(I)=0 ,
     * the corresponding angle(I) is set to 0.
     * @param x input floating-point array of x-coordinates of 2D vectors.
     * @param y input array of y-coordinates of 2D vectors; it must have the
     * same size and the same type as x.
     * @param angle output array of vector angles; it has the same size and
     * same type as x .
     * @param angleInDegrees when true, the function calculates the angle in
     * degrees, otherwise, they are measured in radians.
     */
    public static void phase(Mat x, Mat y, Mat angle, boolean angleInDegrees) {
        phase_0(x.nativeObj, y.nativeObj, angle.nativeObj, angleInDegrees);
    }

    /**
     * Calculates the rotation angle of 2D vectors.
     *
     * The function cv::phase calculates the rotation angle of each 2D vector that
     * is formed from the corresponding elements of x and y :
     * \(\texttt{angle} (I) =  \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))\)
     *
     * The angle estimation accuracy is about 0.3 degrees. When x(I)=y(I)=0 ,
     * the corresponding angle(I) is set to 0.
     * @param x input floating-point array of x-coordinates of 2D vectors.
     * @param y input array of y-coordinates of 2D vectors; it must have the
     * same size and the same type as x.
     * @param angle output array of vector angles; it has the same size and
     * same type as x .
     * degrees, otherwise, they are measured in radians.
     */
    public static void phase(Mat x, Mat y, Mat angle) {
        phase_1(x.nativeObj, y.nativeObj, angle.nativeObj);
    }


    //
    // C++:  void cv::polarToCart(Mat magnitude, Mat angle, Mat& x, Mat& y, bool angleInDegrees = false)
    //

    /**
     * Calculates x and y coordinates of 2D vectors from their magnitude and angle.
     *
     * The function cv::polarToCart calculates the Cartesian coordinates of each 2D
     * vector represented by the corresponding elements of magnitude and angle:
     * \(\begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}\)
     *
     * The relative accuracy of the estimated coordinates is about 1e-6.
     * @param magnitude input floating-point array of magnitudes of 2D vectors;
     * it can be an empty matrix (=Mat()), in this case, the function assumes
     * that all the magnitudes are =1; if it is not empty, it must have the
     * same size and type as angle.
     * @param angle input floating-point array of angles of 2D vectors.
     * @param x output array of x-coordinates of 2D vectors; it has the same
     * size and type as angle.
     * @param y output array of y-coordinates of 2D vectors; it has the same
     * size and type as angle.
     * @param angleInDegrees when true, the input angles are measured in
     * degrees, otherwise, they are measured in radians.
     * SEE: cartToPolar, magnitude, phase, exp, log, pow, sqrt
     */
    public static void polarToCart(Mat magnitude, Mat angle, Mat x, Mat y, boolean angleInDegrees) {
        polarToCart_0(magnitude.nativeObj, angle.nativeObj, x.nativeObj, y.nativeObj, angleInDegrees);
    }

    /**
     * Calculates x and y coordinates of 2D vectors from their magnitude and angle.
     *
     * The function cv::polarToCart calculates the Cartesian coordinates of each 2D
     * vector represented by the corresponding elements of magnitude and angle:
     * \(\begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}\)
     *
     * The relative accuracy of the estimated coordinates is about 1e-6.
     * @param magnitude input floating-point array of magnitudes of 2D vectors;
     * it can be an empty matrix (=Mat()), in this case, the function assumes
     * that all the magnitudes are =1; if it is not empty, it must have the
     * same size and type as angle.
     * @param angle input floating-point array of angles of 2D vectors.
     * @param x output array of x-coordinates of 2D vectors; it has the same
     * size and type as angle.
     * @param y output array of y-coordinates of 2D vectors; it has the same
     * size and type as angle.
     * degrees, otherwise, they are measured in radians.
     * SEE: cartToPolar, magnitude, phase, exp, log, pow, sqrt
     */
    public static void polarToCart(Mat magnitude, Mat angle, Mat x, Mat y) {
        polarToCart_1(magnitude.nativeObj, angle.nativeObj, x.nativeObj, y.nativeObj);
    }


    //
    // C++:  void cv::pow(Mat src, double power, Mat& dst)
    //

    /**
     * Raises every array element to a power.
     *
     * The function cv::pow raises every element of the input array to power :
     * \(\texttt{dst} (I) =  \fork{\texttt{src}(I)^{power}}{if \(\texttt{power}\) is integer}{|\texttt{src}(I)|^{power}}{otherwise}\)
     *
     * So, for a non-integer power exponent, the absolute values of input array
     * elements are used. However, it is possible to get true values for
     * negative values using some extra operations. In the example below,
     * computing the 5th root of array src shows:
     * <code>
     *     Mat mask = src &lt; 0;
     *     pow(src, 1./5, dst);
     *     subtract(Scalar::all(0), dst, dst, mask);
     * </code>
     * For some values of power, such as integer values, 0.5 and -0.5,
     * specialized faster algorithms are used.
     *
     * Special values (NaN, Inf) are not handled.
     * @param src input array.
     * @param power exponent of power.
     * @param dst output array of the same size and type as src.
     * SEE: sqrt, exp, log, cartToPolar, polarToCart
     */
    public static void pow(Mat src, double power, Mat dst) {
        pow_0(src.nativeObj, power, dst.nativeObj);
    }


    //
    // C++:  void cv::randShuffle(Mat& dst, double iterFactor = 1., RNG* rng = 0)
    //

    /**
     * Shuffles the array elements randomly.
     *
     * The function cv::randShuffle shuffles the specified 1D array by randomly choosing pairs of elements and
     * swapping them. The number of such swap operations will be dst.rows\*dst.cols\*iterFactor .
     * @param dst input/output numerical 1D array.
     * @param iterFactor scale factor that determines the number of random swap operations (see the details
     * below).
     * instead.
     * SEE: RNG, sort
     */
    public static void randShuffle(Mat dst, double iterFactor) {
        randShuffle_0(dst.nativeObj, iterFactor);
    }

    /**
     * Shuffles the array elements randomly.
     *
     * The function cv::randShuffle shuffles the specified 1D array by randomly choosing pairs of elements and
     * swapping them. The number of such swap operations will be dst.rows\*dst.cols\*iterFactor .
     * @param dst input/output numerical 1D array.
     * below).
     * instead.
     * SEE: RNG, sort
     */
    public static void randShuffle(Mat dst) {
        randShuffle_2(dst.nativeObj);
    }


    //
    // C++:  void cv::randn(Mat& dst, double mean, double stddev)
    //

    /**
     * Fills the array with normally distributed random numbers.
     *
     * The function cv::randn fills the matrix dst with normally distributed random numbers with the specified
     * mean vector and the standard deviation matrix. The generated random numbers are clipped to fit the
     * value range of the output array data type.
     * @param dst output array of random numbers; the array must be pre-allocated and have 1 to 4 channels.
     * @param mean mean value (expectation) of the generated random numbers.
     * @param stddev standard deviation of the generated random numbers; it can be either a vector (in
     * which case a diagonal standard deviation matrix is assumed) or a square matrix.
     * SEE: RNG, randu
     */
    public static void randn(Mat dst, double mean, double stddev) {
        randn_0(dst.nativeObj, mean, stddev);
    }


    //
    // C++:  void cv::randu(Mat& dst, double low, double high)
    //

    /**
     * Generates a single uniformly-distributed random number or an array of random numbers.
     *
     * Non-template variant of the function fills the matrix dst with uniformly-distributed
     * random numbers from the specified range:
     * \(\texttt{low} _c  \leq \texttt{dst} (I)_c &lt;  \texttt{high} _c\)
     * @param dst output array of random numbers; the array must be pre-allocated.
     * @param low inclusive lower boundary of the generated random numbers.
     * @param high exclusive upper boundary of the generated random numbers.
     * SEE: RNG, randn, theRNG
     */
    public static void randu(Mat dst, double low, double high) {
        randu_0(dst.nativeObj, low, high);
    }


    //
    // C++:  void cv::reduce(Mat src, Mat& dst, int dim, int rtype, int dtype = -1)
    //

    /**
     * Reduces a matrix to a vector.
     *
     * The function #reduce reduces the matrix to a vector by treating the matrix rows/columns as a set of
     * 1D vectors and performing the specified operation on the vectors until a single row/column is
     * obtained. For example, the function can be used to compute horizontal and vertical projections of a
     * raster image. In case of #REDUCE_MAX and #REDUCE_MIN , the output image should have the same type as the source one.
     * In case of #REDUCE_SUM and #REDUCE_AVG , the output may have a larger element bit-depth to preserve accuracy.
     * And multi-channel arrays are also supported in these two reduction modes.
     *
     * The following code demonstrates its usage for a single channel matrix.
     * SNIPPET: snippets/core_reduce.cpp example
     *
     * And the following code demonstrates its usage for a two-channel matrix.
     * SNIPPET: snippets/core_reduce.cpp example2
     *
     * @param src input 2D matrix.
     * @param dst output vector. Its size and type is defined by dim and dtype parameters.
     * @param dim dimension index along which the matrix is reduced. 0 means that the matrix is reduced to
     * a single row. 1 means that the matrix is reduced to a single column.
     * @param rtype reduction operation that could be one of #ReduceTypes
     * @param dtype when negative, the output vector will have the same type as the input matrix,
     * otherwise, its type will be CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()).
     * SEE: repeat
     */
    public static void reduce(Mat src, Mat dst, int dim, int rtype, int dtype) {
        reduce_0(src.nativeObj, dst.nativeObj, dim, rtype, dtype);
    }

    /**
     * Reduces a matrix to a vector.
     *
     * The function #reduce reduces the matrix to a vector by treating the matrix rows/columns as a set of
     * 1D vectors and performing the specified operation on the vectors until a single row/column is
     * obtained. For example, the function can be used to compute horizontal and vertical projections of a
     * raster image. In case of #REDUCE_MAX and #REDUCE_MIN , the output image should have the same type as the source one.
     * In case of #REDUCE_SUM and #REDUCE_AVG , the output may have a larger element bit-depth to preserve accuracy.
     * And multi-channel arrays are also supported in these two reduction modes.
     *
     * The following code demonstrates its usage for a single channel matrix.
     * SNIPPET: snippets/core_reduce.cpp example
     *
     * And the following code demonstrates its usage for a two-channel matrix.
     * SNIPPET: snippets/core_reduce.cpp example2
     *
     * @param src input 2D matrix.
     * @param dst output vector. Its size and type is defined by dim and dtype parameters.
     * @param dim dimension index along which the matrix is reduced. 0 means that the matrix is reduced to
     * a single row. 1 means that the matrix is reduced to a single column.
     * @param rtype reduction operation that could be one of #ReduceTypes
     * otherwise, its type will be CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()).
     * SEE: repeat
     */
    public static void reduce(Mat src, Mat dst, int dim, int rtype) {
        reduce_1(src.nativeObj, dst.nativeObj, dim, rtype);
    }


    //
    // C++:  void cv::repeat(Mat src, int ny, int nx, Mat& dst)
    //

    /**
     * Fills the output array with repeated copies of the input array.
     *
     * The function cv::repeat duplicates the input array one or more times along each of the two axes:
     * \(\texttt{dst} _{ij}= \texttt{src} _{i\mod src.rows, \; j\mod src.cols }\)
     * The second variant of the function is more convenient to use with REF: MatrixExpressions.
     * @param src input array to replicate.
     * @param ny Flag to specify how many times the {@code src} is repeated along the
     * vertical axis.
     * @param nx Flag to specify how many times the {@code src} is repeated along the
     * horizontal axis.
     * @param dst output array of the same type as {@code src}.
     * SEE: cv::reduce
     */
    public static void repeat(Mat src, int ny, int nx, Mat dst) {
        repeat_0(src.nativeObj, ny, nx, dst.nativeObj);
    }


    //
    // C++:  void cv::rotate(Mat src, Mat& dst, int rotateCode)
    //

    /**
     * Rotates a 2D array in multiples of 90 degrees.
     * The function cv::rotate rotates the array in one of three different ways:
     * Rotate by 90 degrees clockwise (rotateCode = ROTATE_90_CLOCKWISE).
     * Rotate by 180 degrees clockwise (rotateCode = ROTATE_180).
     * Rotate by 270 degrees clockwise (rotateCode = ROTATE_90_COUNTERCLOCKWISE).
     * @param src input array.
     * @param dst output array of the same type as src.  The size is the same with ROTATE_180,
     * and the rows and cols are switched for ROTATE_90_CLOCKWISE and ROTATE_90_COUNTERCLOCKWISE.
     * @param rotateCode an enum to specify how to rotate the array; see the enum #RotateFlags
     * SEE: transpose , repeat , completeSymm, flip, RotateFlags
     */
    public static void rotate(Mat src, Mat dst, int rotateCode) {
        rotate_0(src.nativeObj, dst.nativeObj, rotateCode);
    }


    //
    // C++:  void cv::scaleAdd(Mat src1, double alpha, Mat src2, Mat& dst)
    //

    /**
     * Calculates the sum of a scaled array and another array.
     *
     * The function scaleAdd is one of the classical primitive linear algebra operations, known as DAXPY
     * or SAXPY in [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). It calculates
     * the sum of a scaled array and another array:
     * \(\texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)\)
     * The function can also be emulated with a matrix expression, for example:
     * <code>
     *     Mat A(3, 3, CV_64F);
     *     ...
     *     A.row(0) = A.row(1)*2 + A.row(2);
     * </code>
     * @param src1 first input array.
     * @param alpha scale factor for the first array.
     * @param src2 second input array of the same size and type as src1.
     * @param dst output array of the same size and type as src1.
     * SEE: add, addWeighted, subtract, Mat::dot, Mat::convertTo
     */
    public static void scaleAdd(Mat src1, double alpha, Mat src2, Mat dst) {
        scaleAdd_0(src1.nativeObj, alpha, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::setErrorVerbosity(bool verbose)
    //

    public static void setErrorVerbosity(boolean verbose) {
        setErrorVerbosity_0(verbose);
    }


    //
    // C++:  void cv::setIdentity(Mat& mtx, Scalar s = Scalar(1))
    //

    /**
     * Initializes a scaled identity matrix.
     *
     * The function cv::setIdentity initializes a scaled identity matrix:
     * \(\texttt{mtx} (i,j)= \fork{\texttt{value}}{ if \(i=j\)}{0}{otherwise}\)
     *
     * The function can also be emulated using the matrix initializers and the
     * matrix expressions:
     * <code>
     *     Mat A = Mat::eye(4, 3, CV_32F)*5;
     *     // A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]
     * </code>
     * @param mtx matrix to initialize (not necessarily square).
     * @param s value to assign to diagonal elements.
     * SEE: Mat::zeros, Mat::ones, Mat::setTo, Mat::operator=
     */
    public static void setIdentity(Mat mtx, Scalar s) {
        setIdentity_0(mtx.nativeObj, s.val[0], s.val[1], s.val[2], s.val[3]);
    }

    /**
     * Initializes a scaled identity matrix.
     *
     * The function cv::setIdentity initializes a scaled identity matrix:
     * \(\texttt{mtx} (i,j)= \fork{\texttt{value}}{ if \(i=j\)}{0}{otherwise}\)
     *
     * The function can also be emulated using the matrix initializers and the
     * matrix expressions:
     * <code>
     *     Mat A = Mat::eye(4, 3, CV_32F)*5;
     *     // A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]
     * </code>
     * @param mtx matrix to initialize (not necessarily square).
     * SEE: Mat::zeros, Mat::ones, Mat::setTo, Mat::operator=
     */
    public static void setIdentity(Mat mtx) {
        setIdentity_1(mtx.nativeObj);
    }


    //
    // C++:  void cv::setNumThreads(int nthreads)
    //

    /**
     * OpenCV will try to set the number of threads for the next parallel region.
     *
     * If threads == 0, OpenCV will disable threading optimizations and run all it's functions
     * sequentially. Passing threads &lt; 0 will reset threads number to system default. This function must
     * be called outside of parallel region.
     *
     * OpenCV will try to run its functions with specified threads number, but some behaviour differs from
     * framework:
     * <ul>
     *   <li>
     *    {@code TBB} - User-defined parallel constructions will run with the same threads number, if
     *     another is not specified. If later on user creates his own scheduler, OpenCV will use it.
     *   </li>
     *   <li>
     *    {@code OpenMP} - No special defined behaviour.
     *   </li>
     *   <li>
     *    {@code Concurrency} - If threads == 1, OpenCV will disable threading optimizations and run its
     *     functions sequentially.
     *   </li>
     *   <li>
     *    {@code GCD} - Supports only values &lt;= 0.
     *   </li>
     *   <li>
     *    {@code C=} - No special defined behaviour.
     * @param nthreads Number of threads used by OpenCV.
     * SEE: getNumThreads, getThreadNum
     *   </li>
     * </ul>
     */
    public static void setNumThreads(int nthreads) {
        setNumThreads_0(nthreads);
    }


    //
    // C++:  void cv::setRNGSeed(int seed)
    //

    /**
     * Sets state of default random number generator.
     *
     * The function cv::setRNGSeed sets state of default random number generator to custom value.
     * @param seed new state for default random number generator
     * SEE: RNG, randu, randn
     */
    public static void setRNGSeed(int seed) {
        setRNGSeed_0(seed);
    }


    //
    // C++:  void cv::sort(Mat src, Mat& dst, int flags)
    //

    /**
     * Sorts each row or each column of a matrix.
     *
     * The function cv::sort sorts each matrix row or each matrix column in
     * ascending or descending order. So you should pass two operation flags to
     * get desired behaviour. If you want to sort matrix rows or columns
     * lexicographically, you can use STL std::sort generic function with the
     * proper comparison predicate.
     *
     * @param src input single-channel array.
     * @param dst output array of the same size and type as src.
     * @param flags operation flags, a combination of #SortFlags
     * SEE: sortIdx, randShuffle
     */
    public static void sort(Mat src, Mat dst, int flags) {
        sort_0(src.nativeObj, dst.nativeObj, flags);
    }


    //
    // C++:  void cv::sortIdx(Mat src, Mat& dst, int flags)
    //

    /**
     * Sorts each row or each column of a matrix.
     *
     * The function cv::sortIdx sorts each matrix row or each matrix column in the
     * ascending or descending order. So you should pass two operation flags to
     * get desired behaviour. Instead of reordering the elements themselves, it
     * stores the indices of sorted elements in the output array. For example:
     * <code>
     *     Mat A = Mat::eye(3,3,CV_32F), B;
     *     sortIdx(A, B, SORT_EVERY_ROW + SORT_ASCENDING);
     *     // B will probably contain
     *     // (because of equal elements in A some permutations are possible):
     *     // [[1, 2, 0], [0, 2, 1], [0, 1, 2]]
     * </code>
     * @param src input single-channel array.
     * @param dst output integer array of the same size as src.
     * @param flags operation flags that could be a combination of cv::SortFlags
     * SEE: sort, randShuffle
     */
    public static void sortIdx(Mat src, Mat dst, int flags) {
        sortIdx_0(src.nativeObj, dst.nativeObj, flags);
    }


    //
    // C++:  void cv::split(Mat m, vector_Mat& mv)
    //

    /**
     *
     * @param m input multi-channel array.
     * @param mv output vector of arrays; the arrays themselves are reallocated, if needed.
     */
    public static void split(Mat m, List<Mat> mv) {
        Mat mv_mat = new Mat();
        split_0(m.nativeObj, mv_mat.nativeObj);
        Converters.Mat_to_vector_Mat(mv_mat, mv);
        mv_mat.release();
    }


    //
    // C++:  void cv::sqrt(Mat src, Mat& dst)
    //

    /**
     * Calculates a square root of array elements.
     *
     * The function cv::sqrt calculates a square root of each input array element.
     * In case of multi-channel arrays, each channel is processed
     * independently. The accuracy is approximately the same as of the built-in
     * std::sqrt .
     * @param src input floating-point array.
     * @param dst output array of the same size and type as src.
     */
    public static void sqrt(Mat src, Mat dst) {
        sqrt_0(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::subtract(Mat src1, Mat src2, Mat& dst, Mat mask = Mat(), int dtype = -1)
    //

    /**
     * Calculates the per-element difference between two arrays or array and a scalar.
     *
     * The function subtract calculates:
     * <ul>
     *   <li>
     *  Difference between two arrays, when both input arrays have the same size and the same number of
     * channels:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Difference between an array and a scalar, when src2 is constructed from Scalar or has the same
     * number of elements as {@code src1.channels()}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Difference between a scalar and an array, when src1 is constructed from Scalar or has the same
     * number of elements as {@code src2.channels()}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  The reverse difference between a scalar and an array in the case of {@code SubRS}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0\)
     * where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
     * channel is processed independently.
     *   </li>
     * </ul>
     *
     * The first function in the list above can be replaced with matrix expressions:
     * <code>
     *     dst = src1 - src2;
     *     dst -= src1; // equivalent to subtract(dst, src1, dst);
     * </code>
     * The input arrays and the output array can all have the same or different depths. For example, you
     * can subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of
     * the output array is determined by dtype parameter. In the second and third cases above, as well as
     * in the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this
     * case the output array will have the same depth as the input array, be it src1, src2 or both.
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array of the same size and the same number of channels as the input array.
     * @param mask optional operation mask; this is an 8-bit single channel array that specifies elements
     * of the output array to be changed.
     * @param dtype optional depth of the output array
     * SEE:  add, addWeighted, scaleAdd, Mat::convertTo
     */
    public static void subtract(Mat src1, Mat src2, Mat dst, Mat mask, int dtype) {
        subtract_0(src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj, dtype);
    }

    /**
     * Calculates the per-element difference between two arrays or array and a scalar.
     *
     * The function subtract calculates:
     * <ul>
     *   <li>
     *  Difference between two arrays, when both input arrays have the same size and the same number of
     * channels:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Difference between an array and a scalar, when src2 is constructed from Scalar or has the same
     * number of elements as {@code src1.channels()}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Difference between a scalar and an array, when src1 is constructed from Scalar or has the same
     * number of elements as {@code src2.channels()}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  The reverse difference between a scalar and an array in the case of {@code SubRS}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0\)
     * where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
     * channel is processed independently.
     *   </li>
     * </ul>
     *
     * The first function in the list above can be replaced with matrix expressions:
     * <code>
     *     dst = src1 - src2;
     *     dst -= src1; // equivalent to subtract(dst, src1, dst);
     * </code>
     * The input arrays and the output array can all have the same or different depths. For example, you
     * can subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of
     * the output array is determined by dtype parameter. In the second and third cases above, as well as
     * in the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this
     * case the output array will have the same depth as the input array, be it src1, src2 or both.
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array of the same size and the same number of channels as the input array.
     * @param mask optional operation mask; this is an 8-bit single channel array that specifies elements
     * of the output array to be changed.
     * SEE:  add, addWeighted, scaleAdd, Mat::convertTo
     */
    public static void subtract(Mat src1, Mat src2, Mat dst, Mat mask) {
        subtract_1(src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj);
    }

    /**
     * Calculates the per-element difference between two arrays or array and a scalar.
     *
     * The function subtract calculates:
     * <ul>
     *   <li>
     *  Difference between two arrays, when both input arrays have the same size and the same number of
     * channels:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Difference between an array and a scalar, when src2 is constructed from Scalar or has the same
     * number of elements as {@code src1.channels()}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  Difference between a scalar and an array, when src1 is constructed from Scalar or has the same
     * number of elements as {@code src2.channels()}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\)
     *   </li>
     *   <li>
     *  The reverse difference between a scalar and an array in the case of {@code SubRS}:
     *     \(\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0\)
     * where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
     * channel is processed independently.
     *   </li>
     * </ul>
     *
     * The first function in the list above can be replaced with matrix expressions:
     * <code>
     *     dst = src1 - src2;
     *     dst -= src1; // equivalent to subtract(dst, src1, dst);
     * </code>
     * The input arrays and the output array can all have the same or different depths. For example, you
     * can subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of
     * the output array is determined by dtype parameter. In the second and third cases above, as well as
     * in the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this
     * case the output array will have the same depth as the input array, be it src1, src2 or both.
     * <b>Note:</b> Saturation is not applied when the output array has the depth CV_32S. You may even get
     * result of an incorrect sign in the case of overflow.
     * @param src1 first input array or a scalar.
     * @param src2 second input array or a scalar.
     * @param dst output array of the same size and the same number of channels as the input array.
     * of the output array to be changed.
     * SEE:  add, addWeighted, scaleAdd, Mat::convertTo
     */
    public static void subtract(Mat src1, Mat src2, Mat dst) {
        subtract_2(src1.nativeObj, src2.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::subtract(Mat src1, Scalar src2, Mat& dst, Mat mask = Mat(), int dtype = -1)
    //

    public static void subtract(Mat src1, Scalar src2, Mat dst, Mat mask, int dtype) {
        subtract_3(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, mask.nativeObj, dtype);
    }

    public static void subtract(Mat src1, Scalar src2, Mat dst, Mat mask) {
        subtract_4(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj, mask.nativeObj);
    }

    public static void subtract(Mat src1, Scalar src2, Mat dst) {
        subtract_5(src1.nativeObj, src2.val[0], src2.val[1], src2.val[2], src2.val[3], dst.nativeObj);
    }


    //
    // C++:  void cv::transform(Mat src, Mat& dst, Mat m)
    //

    /**
     * Performs the matrix transformation of every array element.
     *
     * The function cv::transform performs the matrix transformation of every
     * element of the array src and stores the results in dst :
     * \(\texttt{dst} (I) =  \texttt{m} \cdot \texttt{src} (I)\)
     * (when m.cols=src.channels() ), or
     * \(\texttt{dst} (I) =  \texttt{m} \cdot [ \texttt{src} (I); 1]\)
     * (when m.cols=src.channels()+1 )
     *
     * Every element of the N -channel array src is interpreted as N -element
     * vector that is transformed using the M x N or M x (N+1) matrix m to
     * M-element vector - the corresponding element of the output array dst .
     *
     * The function may be used for geometrical transformation of
     * N -dimensional points, arbitrary linear color space transformation (such
     * as various kinds of RGB to YUV transforms), shuffling the image
     * channels, and so forth.
     * @param src input array that must have as many channels (1 to 4) as
     * m.cols or m.cols-1.
     * @param dst output array of the same size and depth as src; it has as
     * many channels as m.rows.
     * @param m transformation 2x2 or 2x3 floating-point matrix.
     * SEE: perspectiveTransform, getAffineTransform, estimateAffine2D, warpAffine, warpPerspective
     */
    public static void transform(Mat src, Mat dst, Mat m) {
        transform_0(src.nativeObj, dst.nativeObj, m.nativeObj);
    }


    //
    // C++:  void cv::transpose(Mat src, Mat& dst)
    //

    /**
     * Transposes a matrix.
     *
     * The function cv::transpose transposes the matrix src :
     * \(\texttt{dst} (i,j) =  \texttt{src} (j,i)\)
     * <b>Note:</b> No complex conjugation is done in case of a complex matrix. It
     * should be done separately if needed.
     * @param src input array.
     * @param dst output array of the same type as src.
     */
    public static void transpose(Mat src, Mat dst) {
        transpose_0(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::vconcat(vector_Mat src, Mat& dst)
    //

    /**
     *
     *  <code>
     *     std::vector&lt;cv::Mat&gt; matrices = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
     *                                       cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
     *                                       cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};
     *
     *     cv::Mat out;
     *     cv::vconcat( matrices, out );
     *     //out:
     *     //[1,   1,   1,   1;
     *     // 2,   2,   2,   2;
     *     // 3,   3,   3,   3]
     *  </code>
     *  @param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth
     *  @param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.
     * same depth.
     */
    public static void vconcat(List<Mat> src, Mat dst) {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        vconcat_0(src_mat.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::ipp::setUseIPP(bool flag)
    //

    public static void setUseIPP(boolean flag) {
        setUseIPP_0(flag);
    }


    //
    // C++:  void cv::ipp::setUseIPP_NotExact(bool flag)
    //

    public static void setUseIPP_NotExact(boolean flag) {
        setUseIPP_NotExact_0(flag);
    }


    //
    // C++:  void cv::samples::addSamplesDataSearchPath(String path)
    //

    /**
     * Override search data path by adding new search location
     *
     * Use this only to override default behavior
     * Passed paths are used in LIFO order.
     *
     * @param path Path to used samples data
     */
    public static void addSamplesDataSearchPath(String path) {
        addSamplesDataSearchPath_0(path);
    }


    //
    // C++:  void cv::samples::addSamplesDataSearchSubDirectory(String subdir)
    //

    /**
     * Append samples search data sub directory
     *
     * General usage is to add OpenCV modules name ({@code &lt;opencv_contrib&gt;/modules/&lt;name&gt;/samples/data} -&gt; {@code &lt;name&gt;/samples/data} + {@code modules/&lt;name&gt;/samples/data}).
     * Passed subdirectories are used in LIFO order.
     *
     * @param subdir samples data sub directory
     */
    public static void addSamplesDataSearchSubDirectory(String subdir) {
        addSamplesDataSearchSubDirectory_0(subdir);
    }

// manual port
public static class MinMaxLocResult {
    public double minVal;
    public double maxVal;
    public Point minLoc;
    public Point maxLoc;


    public MinMaxLocResult() {
        minVal=0; maxVal=0;
        minLoc=new Point();
        maxLoc=new Point();
    }
}


// C++: minMaxLoc(Mat src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray())


//javadoc: minMaxLoc(src, mask)
public static MinMaxLocResult minMaxLoc(Mat src, Mat mask) {
    MinMaxLocResult res = new MinMaxLocResult();
    long maskNativeObj=0;
    if (mask != null) {
        maskNativeObj=mask.nativeObj;
    }
    double resarr[] = n_minMaxLocManual(src.nativeObj, maskNativeObj);
    res.minVal=resarr[0];
    res.maxVal=resarr[1];
    res.minLoc.x=resarr[2];
    res.minLoc.y=resarr[3];
    res.maxLoc.x=resarr[4];
    res.maxLoc.y=resarr[5];
    return res;
}


//javadoc: minMaxLoc(src)
public static MinMaxLocResult minMaxLoc(Mat src) {
    return minMaxLoc(src, null);
}


    // C++:  Scalar cv::mean(Mat src, Mat mask = Mat())
    private static native double[] mean_0(long src_nativeObj, long mask_nativeObj);
    private static native double[] mean_1(long src_nativeObj);

    // C++:  Scalar cv::sum(Mat src)
    private static native double[] sumElems_0(long src_nativeObj);

    // C++:  Scalar cv::trace(Mat mtx)
    private static native double[] trace_0(long mtx_nativeObj);

    // C++:  String cv::getBuildInformation()
    private static native String getBuildInformation_0();

    // C++:  String cv::getHardwareFeatureName(int feature)
    private static native String getHardwareFeatureName_0(int feature);

    // C++:  String cv::getVersionString()
    private static native String getVersionString_0();

    // C++:  String cv::ipp::getIppVersion()
    private static native String getIppVersion_0();

    // C++:  String cv::samples::findFile(String relative_path, bool required = true, bool silentMode = false)
    private static native String findFile_0(String relative_path, boolean required, boolean silentMode);
    private static native String findFile_1(String relative_path, boolean required);
    private static native String findFile_2(String relative_path);

    // C++:  String cv::samples::findFileOrKeep(String relative_path, bool silentMode = false)
    private static native String findFileOrKeep_0(String relative_path, boolean silentMode);
    private static native String findFileOrKeep_1(String relative_path);

    // C++:  bool cv::checkRange(Mat a, bool quiet = true,  _hidden_ * pos = 0, double minVal = -DBL_MAX, double maxVal = DBL_MAX)
    private static native boolean checkRange_0(long a_nativeObj, boolean quiet, double minVal, double maxVal);
    private static native boolean checkRange_1(long a_nativeObj, boolean quiet, double minVal);
    private static native boolean checkRange_2(long a_nativeObj, boolean quiet);
    private static native boolean checkRange_4(long a_nativeObj);

    // C++:  bool cv::eigen(Mat src, Mat& eigenvalues, Mat& eigenvectors = Mat())
    private static native boolean eigen_0(long src_nativeObj, long eigenvalues_nativeObj, long eigenvectors_nativeObj);
    private static native boolean eigen_1(long src_nativeObj, long eigenvalues_nativeObj);

    // C++:  bool cv::solve(Mat src1, Mat src2, Mat& dst, int flags = DECOMP_LU)
    private static native boolean solve_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, int flags);
    private static native boolean solve_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  bool cv::ipp::useIPP()
    private static native boolean useIPP_0();

    // C++:  bool cv::ipp::useIPP_NotExact()
    private static native boolean useIPP_NotExact_0();

    // C++:  double cv::Mahalanobis(Mat v1, Mat v2, Mat icovar)
    private static native double Mahalanobis_0(long v1_nativeObj, long v2_nativeObj, long icovar_nativeObj);

    // C++:  double cv::PSNR(Mat src1, Mat src2, double R = 255.)
    private static native double PSNR_0(long src1_nativeObj, long src2_nativeObj, double R);
    private static native double PSNR_1(long src1_nativeObj, long src2_nativeObj);

    // C++:  double cv::determinant(Mat mtx)
    private static native double determinant_0(long mtx_nativeObj);

    // C++:  double cv::getTickFrequency()
    private static native double getTickFrequency_0();

    // C++:  double cv::invert(Mat src, Mat& dst, int flags = DECOMP_LU)
    private static native double invert_0(long src_nativeObj, long dst_nativeObj, int flags);
    private static native double invert_1(long src_nativeObj, long dst_nativeObj);

    // C++:  double cv::kmeans(Mat data, int K, Mat& bestLabels, TermCriteria criteria, int attempts, int flags, Mat& centers = Mat())
    private static native double kmeans_0(long data_nativeObj, int K, long bestLabels_nativeObj, int criteria_type, int criteria_maxCount, double criteria_epsilon, int attempts, int flags, long centers_nativeObj);
    private static native double kmeans_1(long data_nativeObj, int K, long bestLabels_nativeObj, int criteria_type, int criteria_maxCount, double criteria_epsilon, int attempts, int flags);

    // C++:  double cv::norm(Mat src1, Mat src2, int normType = NORM_L2, Mat mask = Mat())
    private static native double norm_0(long src1_nativeObj, long src2_nativeObj, int normType, long mask_nativeObj);
    private static native double norm_1(long src1_nativeObj, long src2_nativeObj, int normType);
    private static native double norm_2(long src1_nativeObj, long src2_nativeObj);

    // C++:  double cv::norm(Mat src1, int normType = NORM_L2, Mat mask = Mat())
    private static native double norm_3(long src1_nativeObj, int normType, long mask_nativeObj);
    private static native double norm_4(long src1_nativeObj, int normType);
    private static native double norm_5(long src1_nativeObj);

    // C++:  double cv::solvePoly(Mat coeffs, Mat& roots, int maxIters = 300)
    private static native double solvePoly_0(long coeffs_nativeObj, long roots_nativeObj, int maxIters);
    private static native double solvePoly_1(long coeffs_nativeObj, long roots_nativeObj);

    // C++:  float cv::cubeRoot(float val)
    private static native float cubeRoot_0(float val);

    // C++:  float cv::fastAtan2(float y, float x)
    private static native float fastAtan2_0(float y, float x);

    // C++:  int cv::borderInterpolate(int p, int len, int borderType)
    private static native int borderInterpolate_0(int p, int len, int borderType);

    // C++:  int cv::countNonZero(Mat src)
    private static native int countNonZero_0(long src_nativeObj);

    // C++:  int cv::getNumThreads()
    private static native int getNumThreads_0();

    // C++:  int cv::getNumberOfCPUs()
    private static native int getNumberOfCPUs_0();

    // C++:  int cv::getOptimalDFTSize(int vecsize)
    private static native int getOptimalDFTSize_0(int vecsize);

    // C++:  int cv::getThreadNum()
    private static native int getThreadNum_0();

    // C++:  int cv::getVersionMajor()
    private static native int getVersionMajor_0();

    // C++:  int cv::getVersionMinor()
    private static native int getVersionMinor_0();

    // C++:  int cv::getVersionRevision()
    private static native int getVersionRevision_0();

    // C++:  int cv::solveCubic(Mat coeffs, Mat& roots)
    private static native int solveCubic_0(long coeffs_nativeObj, long roots_nativeObj);

    // C++:  int64 cv::getCPUTickCount()
    private static native long getCPUTickCount_0();

    // C++:  int64 cv::getTickCount()
    private static native long getTickCount_0();

    // C++:  string cv::getCPUFeaturesLine()
    private static native String getCPUFeaturesLine_0();

    // C++:  void cv::LUT(Mat src, Mat lut, Mat& dst)
    private static native void LUT_0(long src_nativeObj, long lut_nativeObj, long dst_nativeObj);

    // C++:  void cv::PCABackProject(Mat data, Mat mean, Mat eigenvectors, Mat& result)
    private static native void PCABackProject_0(long data_nativeObj, long mean_nativeObj, long eigenvectors_nativeObj, long result_nativeObj);

    // C++:  void cv::PCACompute(Mat data, Mat& mean, Mat& eigenvectors, Mat& eigenvalues, double retainedVariance)
    private static native void PCACompute2_0(long data_nativeObj, long mean_nativeObj, long eigenvectors_nativeObj, long eigenvalues_nativeObj, double retainedVariance);

    // C++:  void cv::PCACompute(Mat data, Mat& mean, Mat& eigenvectors, Mat& eigenvalues, int maxComponents = 0)
    private static native void PCACompute2_1(long data_nativeObj, long mean_nativeObj, long eigenvectors_nativeObj, long eigenvalues_nativeObj, int maxComponents);
    private static native void PCACompute2_2(long data_nativeObj, long mean_nativeObj, long eigenvectors_nativeObj, long eigenvalues_nativeObj);

    // C++:  void cv::PCACompute(Mat data, Mat& mean, Mat& eigenvectors, double retainedVariance)
    private static native void PCACompute_0(long data_nativeObj, long mean_nativeObj, long eigenvectors_nativeObj, double retainedVariance);

    // C++:  void cv::PCACompute(Mat data, Mat& mean, Mat& eigenvectors, int maxComponents = 0)
    private static native void PCACompute_1(long data_nativeObj, long mean_nativeObj, long eigenvectors_nativeObj, int maxComponents);
    private static native void PCACompute_2(long data_nativeObj, long mean_nativeObj, long eigenvectors_nativeObj);

    // C++:  void cv::PCAProject(Mat data, Mat mean, Mat eigenvectors, Mat& result)
    private static native void PCAProject_0(long data_nativeObj, long mean_nativeObj, long eigenvectors_nativeObj, long result_nativeObj);

    // C++:  void cv::SVBackSubst(Mat w, Mat u, Mat vt, Mat rhs, Mat& dst)
    private static native void SVBackSubst_0(long w_nativeObj, long u_nativeObj, long vt_nativeObj, long rhs_nativeObj, long dst_nativeObj);

    // C++:  void cv::SVDecomp(Mat src, Mat& w, Mat& u, Mat& vt, int flags = 0)
    private static native void SVDecomp_0(long src_nativeObj, long w_nativeObj, long u_nativeObj, long vt_nativeObj, int flags);
    private static native void SVDecomp_1(long src_nativeObj, long w_nativeObj, long u_nativeObj, long vt_nativeObj);

    // C++:  void cv::absdiff(Mat src1, Mat src2, Mat& dst)
    private static native void absdiff_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::absdiff(Mat src1, Scalar src2, Mat& dst)
    private static native void absdiff_1(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj);

    // C++:  void cv::add(Mat src1, Mat src2, Mat& dst, Mat mask = Mat(), int dtype = -1)
    private static native void add_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, long mask_nativeObj, int dtype);
    private static native void add_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void add_2(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::add(Mat src1, Scalar src2, Mat& dst, Mat mask = Mat(), int dtype = -1)
    private static native void add_3(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, long mask_nativeObj, int dtype);
    private static native void add_4(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, long mask_nativeObj);
    private static native void add_5(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj);

    // C++:  void cv::addWeighted(Mat src1, double alpha, Mat src2, double beta, double gamma, Mat& dst, int dtype = -1)
    private static native void addWeighted_0(long src1_nativeObj, double alpha, long src2_nativeObj, double beta, double gamma, long dst_nativeObj, int dtype);
    private static native void addWeighted_1(long src1_nativeObj, double alpha, long src2_nativeObj, double beta, double gamma, long dst_nativeObj);

    // C++:  void cv::batchDistance(Mat src1, Mat src2, Mat& dist, int dtype, Mat& nidx, int normType = NORM_L2, int K = 0, Mat mask = Mat(), int update = 0, bool crosscheck = false)
    private static native void batchDistance_0(long src1_nativeObj, long src2_nativeObj, long dist_nativeObj, int dtype, long nidx_nativeObj, int normType, int K, long mask_nativeObj, int update, boolean crosscheck);
    private static native void batchDistance_1(long src1_nativeObj, long src2_nativeObj, long dist_nativeObj, int dtype, long nidx_nativeObj, int normType, int K, long mask_nativeObj, int update);
    private static native void batchDistance_2(long src1_nativeObj, long src2_nativeObj, long dist_nativeObj, int dtype, long nidx_nativeObj, int normType, int K, long mask_nativeObj);
    private static native void batchDistance_3(long src1_nativeObj, long src2_nativeObj, long dist_nativeObj, int dtype, long nidx_nativeObj, int normType, int K);
    private static native void batchDistance_4(long src1_nativeObj, long src2_nativeObj, long dist_nativeObj, int dtype, long nidx_nativeObj, int normType);
    private static native void batchDistance_5(long src1_nativeObj, long src2_nativeObj, long dist_nativeObj, int dtype, long nidx_nativeObj);

    // C++:  void cv::bitwise_and(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
    private static native void bitwise_and_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void bitwise_and_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::bitwise_not(Mat src, Mat& dst, Mat mask = Mat())
    private static native void bitwise_not_0(long src_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void bitwise_not_1(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::bitwise_or(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
    private static native void bitwise_or_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void bitwise_or_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::bitwise_xor(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
    private static native void bitwise_xor_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void bitwise_xor_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::calcCovarMatrix(Mat samples, Mat& covar, Mat& mean, int flags, int ctype = CV_64F)
    private static native void calcCovarMatrix_0(long samples_nativeObj, long covar_nativeObj, long mean_nativeObj, int flags, int ctype);
    private static native void calcCovarMatrix_1(long samples_nativeObj, long covar_nativeObj, long mean_nativeObj, int flags);

    // C++:  void cv::cartToPolar(Mat x, Mat y, Mat& magnitude, Mat& angle, bool angleInDegrees = false)
    private static native void cartToPolar_0(long x_nativeObj, long y_nativeObj, long magnitude_nativeObj, long angle_nativeObj, boolean angleInDegrees);
    private static native void cartToPolar_1(long x_nativeObj, long y_nativeObj, long magnitude_nativeObj, long angle_nativeObj);

    // C++:  void cv::compare(Mat src1, Mat src2, Mat& dst, int cmpop)
    private static native void compare_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, int cmpop);

    // C++:  void cv::compare(Mat src1, Scalar src2, Mat& dst, int cmpop)
    private static native void compare_1(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, int cmpop);

    // C++:  void cv::completeSymm(Mat& m, bool lowerToUpper = false)
    private static native void completeSymm_0(long m_nativeObj, boolean lowerToUpper);
    private static native void completeSymm_1(long m_nativeObj);

    // C++:  void cv::convertFp16(Mat src, Mat& dst)
    private static native void convertFp16_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::convertScaleAbs(Mat src, Mat& dst, double alpha = 1, double beta = 0)
    private static native void convertScaleAbs_0(long src_nativeObj, long dst_nativeObj, double alpha, double beta);
    private static native void convertScaleAbs_1(long src_nativeObj, long dst_nativeObj, double alpha);
    private static native void convertScaleAbs_2(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::copyMakeBorder(Mat src, Mat& dst, int top, int bottom, int left, int right, int borderType, Scalar value = Scalar())
    private static native void copyMakeBorder_0(long src_nativeObj, long dst_nativeObj, int top, int bottom, int left, int right, int borderType, double value_val0, double value_val1, double value_val2, double value_val3);
    private static native void copyMakeBorder_1(long src_nativeObj, long dst_nativeObj, int top, int bottom, int left, int right, int borderType);

    // C++:  void cv::copyTo(Mat src, Mat& dst, Mat mask)
    private static native void copyTo_0(long src_nativeObj, long dst_nativeObj, long mask_nativeObj);

    // C++:  void cv::dct(Mat src, Mat& dst, int flags = 0)
    private static native void dct_0(long src_nativeObj, long dst_nativeObj, int flags);
    private static native void dct_1(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::dft(Mat src, Mat& dst, int flags = 0, int nonzeroRows = 0)
    private static native void dft_0(long src_nativeObj, long dst_nativeObj, int flags, int nonzeroRows);
    private static native void dft_1(long src_nativeObj, long dst_nativeObj, int flags);
    private static native void dft_2(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::divide(Mat src1, Mat src2, Mat& dst, double scale = 1, int dtype = -1)
    private static native void divide_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, double scale, int dtype);
    private static native void divide_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, double scale);
    private static native void divide_2(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::divide(Mat src1, Scalar src2, Mat& dst, double scale = 1, int dtype = -1)
    private static native void divide_3(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, double scale, int dtype);
    private static native void divide_4(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, double scale);
    private static native void divide_5(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj);

    // C++:  void cv::divide(double scale, Mat src2, Mat& dst, int dtype = -1)
    private static native void divide_6(double scale, long src2_nativeObj, long dst_nativeObj, int dtype);
    private static native void divide_7(double scale, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::eigenNonSymmetric(Mat src, Mat& eigenvalues, Mat& eigenvectors)
    private static native void eigenNonSymmetric_0(long src_nativeObj, long eigenvalues_nativeObj, long eigenvectors_nativeObj);

    // C++:  void cv::exp(Mat src, Mat& dst)
    private static native void exp_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::extractChannel(Mat src, Mat& dst, int coi)
    private static native void extractChannel_0(long src_nativeObj, long dst_nativeObj, int coi);

    // C++:  void cv::findNonZero(Mat src, Mat& idx)
    private static native void findNonZero_0(long src_nativeObj, long idx_nativeObj);

    // C++:  void cv::flip(Mat src, Mat& dst, int flipCode)
    private static native void flip_0(long src_nativeObj, long dst_nativeObj, int flipCode);

    // C++:  void cv::gemm(Mat src1, Mat src2, double alpha, Mat src3, double beta, Mat& dst, int flags = 0)
    private static native void gemm_0(long src1_nativeObj, long src2_nativeObj, double alpha, long src3_nativeObj, double beta, long dst_nativeObj, int flags);
    private static native void gemm_1(long src1_nativeObj, long src2_nativeObj, double alpha, long src3_nativeObj, double beta, long dst_nativeObj);

    // C++:  void cv::hconcat(vector_Mat src, Mat& dst)
    private static native void hconcat_0(long src_mat_nativeObj, long dst_nativeObj);

    // C++:  void cv::idct(Mat src, Mat& dst, int flags = 0)
    private static native void idct_0(long src_nativeObj, long dst_nativeObj, int flags);
    private static native void idct_1(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::idft(Mat src, Mat& dst, int flags = 0, int nonzeroRows = 0)
    private static native void idft_0(long src_nativeObj, long dst_nativeObj, int flags, int nonzeroRows);
    private static native void idft_1(long src_nativeObj, long dst_nativeObj, int flags);
    private static native void idft_2(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::inRange(Mat src, Scalar lowerb, Scalar upperb, Mat& dst)
    private static native void inRange_0(long src_nativeObj, double lowerb_val0, double lowerb_val1, double lowerb_val2, double lowerb_val3, double upperb_val0, double upperb_val1, double upperb_val2, double upperb_val3, long dst_nativeObj);

    // C++:  void cv::insertChannel(Mat src, Mat& dst, int coi)
    private static native void insertChannel_0(long src_nativeObj, long dst_nativeObj, int coi);

    // C++:  void cv::log(Mat src, Mat& dst)
    private static native void log_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::magnitude(Mat x, Mat y, Mat& magnitude)
    private static native void magnitude_0(long x_nativeObj, long y_nativeObj, long magnitude_nativeObj);

    // C++:  void cv::max(Mat src1, Mat src2, Mat& dst)
    private static native void max_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::max(Mat src1, Scalar src2, Mat& dst)
    private static native void max_1(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj);

    // C++:  void cv::meanStdDev(Mat src, vector_double& mean, vector_double& stddev, Mat mask = Mat())
    private static native void meanStdDev_0(long src_nativeObj, long mean_mat_nativeObj, long stddev_mat_nativeObj, long mask_nativeObj);
    private static native void meanStdDev_1(long src_nativeObj, long mean_mat_nativeObj, long stddev_mat_nativeObj);

    // C++:  void cv::merge(vector_Mat mv, Mat& dst)
    private static native void merge_0(long mv_mat_nativeObj, long dst_nativeObj);

    // C++:  void cv::min(Mat src1, Mat src2, Mat& dst)
    private static native void min_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::min(Mat src1, Scalar src2, Mat& dst)
    private static native void min_1(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj);

    // C++:  void cv::mixChannels(vector_Mat src, vector_Mat dst, vector_int fromTo)
    private static native void mixChannels_0(long src_mat_nativeObj, long dst_mat_nativeObj, long fromTo_mat_nativeObj);

    // C++:  void cv::mulSpectrums(Mat a, Mat b, Mat& c, int flags, bool conjB = false)
    private static native void mulSpectrums_0(long a_nativeObj, long b_nativeObj, long c_nativeObj, int flags, boolean conjB);
    private static native void mulSpectrums_1(long a_nativeObj, long b_nativeObj, long c_nativeObj, int flags);

    // C++:  void cv::mulTransposed(Mat src, Mat& dst, bool aTa, Mat delta = Mat(), double scale = 1, int dtype = -1)
    private static native void mulTransposed_0(long src_nativeObj, long dst_nativeObj, boolean aTa, long delta_nativeObj, double scale, int dtype);
    private static native void mulTransposed_1(long src_nativeObj, long dst_nativeObj, boolean aTa, long delta_nativeObj, double scale);
    private static native void mulTransposed_2(long src_nativeObj, long dst_nativeObj, boolean aTa, long delta_nativeObj);
    private static native void mulTransposed_3(long src_nativeObj, long dst_nativeObj, boolean aTa);

    // C++:  void cv::multiply(Mat src1, Mat src2, Mat& dst, double scale = 1, int dtype = -1)
    private static native void multiply_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, double scale, int dtype);
    private static native void multiply_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, double scale);
    private static native void multiply_2(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::multiply(Mat src1, Scalar src2, Mat& dst, double scale = 1, int dtype = -1)
    private static native void multiply_3(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, double scale, int dtype);
    private static native void multiply_4(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, double scale);
    private static native void multiply_5(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj);

    // C++:  void cv::normalize(Mat src, Mat& dst, double alpha = 1, double beta = 0, int norm_type = NORM_L2, int dtype = -1, Mat mask = Mat())
    private static native void normalize_0(long src_nativeObj, long dst_nativeObj, double alpha, double beta, int norm_type, int dtype, long mask_nativeObj);
    private static native void normalize_1(long src_nativeObj, long dst_nativeObj, double alpha, double beta, int norm_type, int dtype);
    private static native void normalize_2(long src_nativeObj, long dst_nativeObj, double alpha, double beta, int norm_type);
    private static native void normalize_3(long src_nativeObj, long dst_nativeObj, double alpha, double beta);
    private static native void normalize_4(long src_nativeObj, long dst_nativeObj, double alpha);
    private static native void normalize_5(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::patchNaNs(Mat& a, double val = 0)
    private static native void patchNaNs_0(long a_nativeObj, double val);
    private static native void patchNaNs_1(long a_nativeObj);

    // C++:  void cv::perspectiveTransform(Mat src, Mat& dst, Mat m)
    private static native void perspectiveTransform_0(long src_nativeObj, long dst_nativeObj, long m_nativeObj);

    // C++:  void cv::phase(Mat x, Mat y, Mat& angle, bool angleInDegrees = false)
    private static native void phase_0(long x_nativeObj, long y_nativeObj, long angle_nativeObj, boolean angleInDegrees);
    private static native void phase_1(long x_nativeObj, long y_nativeObj, long angle_nativeObj);

    // C++:  void cv::polarToCart(Mat magnitude, Mat angle, Mat& x, Mat& y, bool angleInDegrees = false)
    private static native void polarToCart_0(long magnitude_nativeObj, long angle_nativeObj, long x_nativeObj, long y_nativeObj, boolean angleInDegrees);
    private static native void polarToCart_1(long magnitude_nativeObj, long angle_nativeObj, long x_nativeObj, long y_nativeObj);

    // C++:  void cv::pow(Mat src, double power, Mat& dst)
    private static native void pow_0(long src_nativeObj, double power, long dst_nativeObj);

    // C++:  void cv::randShuffle(Mat& dst, double iterFactor = 1., RNG* rng = 0)
    private static native void randShuffle_0(long dst_nativeObj, double iterFactor);
    private static native void randShuffle_2(long dst_nativeObj);

    // C++:  void cv::randn(Mat& dst, double mean, double stddev)
    private static native void randn_0(long dst_nativeObj, double mean, double stddev);

    // C++:  void cv::randu(Mat& dst, double low, double high)
    private static native void randu_0(long dst_nativeObj, double low, double high);

    // C++:  void cv::reduce(Mat src, Mat& dst, int dim, int rtype, int dtype = -1)
    private static native void reduce_0(long src_nativeObj, long dst_nativeObj, int dim, int rtype, int dtype);
    private static native void reduce_1(long src_nativeObj, long dst_nativeObj, int dim, int rtype);

    // C++:  void cv::repeat(Mat src, int ny, int nx, Mat& dst)
    private static native void repeat_0(long src_nativeObj, int ny, int nx, long dst_nativeObj);

    // C++:  void cv::rotate(Mat src, Mat& dst, int rotateCode)
    private static native void rotate_0(long src_nativeObj, long dst_nativeObj, int rotateCode);

    // C++:  void cv::scaleAdd(Mat src1, double alpha, Mat src2, Mat& dst)
    private static native void scaleAdd_0(long src1_nativeObj, double alpha, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::setErrorVerbosity(bool verbose)
    private static native void setErrorVerbosity_0(boolean verbose);

    // C++:  void cv::setIdentity(Mat& mtx, Scalar s = Scalar(1))
    private static native void setIdentity_0(long mtx_nativeObj, double s_val0, double s_val1, double s_val2, double s_val3);
    private static native void setIdentity_1(long mtx_nativeObj);

    // C++:  void cv::setNumThreads(int nthreads)
    private static native void setNumThreads_0(int nthreads);

    // C++:  void cv::setRNGSeed(int seed)
    private static native void setRNGSeed_0(int seed);

    // C++:  void cv::sort(Mat src, Mat& dst, int flags)
    private static native void sort_0(long src_nativeObj, long dst_nativeObj, int flags);

    // C++:  void cv::sortIdx(Mat src, Mat& dst, int flags)
    private static native void sortIdx_0(long src_nativeObj, long dst_nativeObj, int flags);

    // C++:  void cv::split(Mat m, vector_Mat& mv)
    private static native void split_0(long m_nativeObj, long mv_mat_nativeObj);

    // C++:  void cv::sqrt(Mat src, Mat& dst)
    private static native void sqrt_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::subtract(Mat src1, Mat src2, Mat& dst, Mat mask = Mat(), int dtype = -1)
    private static native void subtract_0(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, long mask_nativeObj, int dtype);
    private static native void subtract_1(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj, long mask_nativeObj);
    private static native void subtract_2(long src1_nativeObj, long src2_nativeObj, long dst_nativeObj);

    // C++:  void cv::subtract(Mat src1, Scalar src2, Mat& dst, Mat mask = Mat(), int dtype = -1)
    private static native void subtract_3(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, long mask_nativeObj, int dtype);
    private static native void subtract_4(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj, long mask_nativeObj);
    private static native void subtract_5(long src1_nativeObj, double src2_val0, double src2_val1, double src2_val2, double src2_val3, long dst_nativeObj);

    // C++:  void cv::transform(Mat src, Mat& dst, Mat m)
    private static native void transform_0(long src_nativeObj, long dst_nativeObj, long m_nativeObj);

    // C++:  void cv::transpose(Mat src, Mat& dst)
    private static native void transpose_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::vconcat(vector_Mat src, Mat& dst)
    private static native void vconcat_0(long src_mat_nativeObj, long dst_nativeObj);

    // C++:  void cv::ipp::setUseIPP(bool flag)
    private static native void setUseIPP_0(boolean flag);

    // C++:  void cv::ipp::setUseIPP_NotExact(bool flag)
    private static native void setUseIPP_NotExact_0(boolean flag);

    // C++:  void cv::samples::addSamplesDataSearchPath(String path)
    private static native void addSamplesDataSearchPath_0(String path);

    // C++:  void cv::samples::addSamplesDataSearchSubDirectory(String subdir)
    private static native void addSamplesDataSearchSubDirectory_0(String subdir);
private static native double[] n_minMaxLocManual(long src_nativeObj, long mask_nativeObj);

}
