//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.photo.AlignMTB;
import org.opencv.photo.CalibrateDebevec;
import org.opencv.photo.CalibrateRobertson;
import org.opencv.photo.MergeDebevec;
import org.opencv.photo.MergeMertens;
import org.opencv.photo.MergeRobertson;
import org.opencv.photo.Tonemap;
import org.opencv.photo.TonemapDrago;
import org.opencv.photo.TonemapMantiuk;
import org.opencv.photo.TonemapReinhard;
import org.opencv.utils.Converters;

// C++: class Photo

public class Photo {

    // C++: enum <unnamed>
    public static final int
            INPAINT_NS = 0,
            INPAINT_TELEA = 1,
            LDR_SIZE = 256,
            NORMAL_CLONE = 1,
            MIXED_CLONE = 2,
            MONOCHROME_TRANSFER = 3,
            RECURS_FILTER = 1,
            NORMCONV_FILTER = 2;


    //
    // C++:  Ptr_AlignMTB cv::createAlignMTB(int max_bits = 6, int exclude_range = 4, bool cut = true)
    //

    /**
     * Creates AlignMTB object
     *
     * @param max_bits logarithm to the base 2 of maximal shift in each dimension. Values of 5 and 6 are
     * usually good enough (31 and 63 pixels shift respectively).
     * @param exclude_range range for exclusion bitmap that is constructed to suppress noise around the
     * median value.
     * @param cut if true cuts images, otherwise fills the new regions with zeros.
     * @return automatically generated
     */
    public static AlignMTB createAlignMTB(int max_bits, int exclude_range, boolean cut) {
        return AlignMTB.__fromPtr__(createAlignMTB_0(max_bits, exclude_range, cut));
    }

    /**
     * Creates AlignMTB object
     *
     * @param max_bits logarithm to the base 2 of maximal shift in each dimension. Values of 5 and 6 are
     * usually good enough (31 and 63 pixels shift respectively).
     * @param exclude_range range for exclusion bitmap that is constructed to suppress noise around the
     * median value.
     * @return automatically generated
     */
    public static AlignMTB createAlignMTB(int max_bits, int exclude_range) {
        return AlignMTB.__fromPtr__(createAlignMTB_1(max_bits, exclude_range));
    }

    /**
     * Creates AlignMTB object
     *
     * @param max_bits logarithm to the base 2 of maximal shift in each dimension. Values of 5 and 6 are
     * usually good enough (31 and 63 pixels shift respectively).
     * median value.
     * @return automatically generated
     */
    public static AlignMTB createAlignMTB(int max_bits) {
        return AlignMTB.__fromPtr__(createAlignMTB_2(max_bits));
    }

    /**
     * Creates AlignMTB object
     *
     * usually good enough (31 and 63 pixels shift respectively).
     * median value.
     * @return automatically generated
     */
    public static AlignMTB createAlignMTB() {
        return AlignMTB.__fromPtr__(createAlignMTB_3());
    }


    //
    // C++:  Ptr_CalibrateDebevec cv::createCalibrateDebevec(int samples = 70, float lambda = 10.0f, bool random = false)
    //

    /**
     * Creates CalibrateDebevec object
     *
     * @param samples number of pixel locations to use
     * @param lambda smoothness term weight. Greater values produce smoother results, but can alter the
     * response.
     * @param random if true sample pixel locations are chosen at random, otherwise they form a
     * rectangular grid.
     * @return automatically generated
     */
    public static CalibrateDebevec createCalibrateDebevec(int samples, float lambda, boolean random) {
        return CalibrateDebevec.__fromPtr__(createCalibrateDebevec_0(samples, lambda, random));
    }

    /**
     * Creates CalibrateDebevec object
     *
     * @param samples number of pixel locations to use
     * @param lambda smoothness term weight. Greater values produce smoother results, but can alter the
     * response.
     * rectangular grid.
     * @return automatically generated
     */
    public static CalibrateDebevec createCalibrateDebevec(int samples, float lambda) {
        return CalibrateDebevec.__fromPtr__(createCalibrateDebevec_1(samples, lambda));
    }

    /**
     * Creates CalibrateDebevec object
     *
     * @param samples number of pixel locations to use
     * response.
     * rectangular grid.
     * @return automatically generated
     */
    public static CalibrateDebevec createCalibrateDebevec(int samples) {
        return CalibrateDebevec.__fromPtr__(createCalibrateDebevec_2(samples));
    }

    /**
     * Creates CalibrateDebevec object
     *
     * response.
     * rectangular grid.
     * @return automatically generated
     */
    public static CalibrateDebevec createCalibrateDebevec() {
        return CalibrateDebevec.__fromPtr__(createCalibrateDebevec_3());
    }


    //
    // C++:  Ptr_CalibrateRobertson cv::createCalibrateRobertson(int max_iter = 30, float threshold = 0.01f)
    //

    /**
     * Creates CalibrateRobertson object
     *
     * @param max_iter maximal number of Gauss-Seidel solver iterations.
     * @param threshold target difference between results of two successive steps of the minimization.
     * @return automatically generated
     */
    public static CalibrateRobertson createCalibrateRobertson(int max_iter, float threshold) {
        return CalibrateRobertson.__fromPtr__(createCalibrateRobertson_0(max_iter, threshold));
    }

    /**
     * Creates CalibrateRobertson object
     *
     * @param max_iter maximal number of Gauss-Seidel solver iterations.
     * @return automatically generated
     */
    public static CalibrateRobertson createCalibrateRobertson(int max_iter) {
        return CalibrateRobertson.__fromPtr__(createCalibrateRobertson_1(max_iter));
    }

    /**
     * Creates CalibrateRobertson object
     *
     * @return automatically generated
     */
    public static CalibrateRobertson createCalibrateRobertson() {
        return CalibrateRobertson.__fromPtr__(createCalibrateRobertson_2());
    }


    //
    // C++:  Ptr_MergeDebevec cv::createMergeDebevec()
    //

    /**
     * Creates MergeDebevec object
     * @return automatically generated
     */
    public static MergeDebevec createMergeDebevec() {
        return MergeDebevec.__fromPtr__(createMergeDebevec_0());
    }


    //
    // C++:  Ptr_MergeMertens cv::createMergeMertens(float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f)
    //

    /**
     * Creates MergeMertens object
     *
     * @param contrast_weight contrast measure weight. See MergeMertens.
     * @param saturation_weight saturation measure weight
     * @param exposure_weight well-exposedness measure weight
     * @return automatically generated
     */
    public static MergeMertens createMergeMertens(float contrast_weight, float saturation_weight, float exposure_weight) {
        return MergeMertens.__fromPtr__(createMergeMertens_0(contrast_weight, saturation_weight, exposure_weight));
    }

    /**
     * Creates MergeMertens object
     *
     * @param contrast_weight contrast measure weight. See MergeMertens.
     * @param saturation_weight saturation measure weight
     * @return automatically generated
     */
    public static MergeMertens createMergeMertens(float contrast_weight, float saturation_weight) {
        return MergeMertens.__fromPtr__(createMergeMertens_1(contrast_weight, saturation_weight));
    }

    /**
     * Creates MergeMertens object
     *
     * @param contrast_weight contrast measure weight. See MergeMertens.
     * @return automatically generated
     */
    public static MergeMertens createMergeMertens(float contrast_weight) {
        return MergeMertens.__fromPtr__(createMergeMertens_2(contrast_weight));
    }

    /**
     * Creates MergeMertens object
     *
     * @return automatically generated
     */
    public static MergeMertens createMergeMertens() {
        return MergeMertens.__fromPtr__(createMergeMertens_3());
    }


    //
    // C++:  Ptr_MergeRobertson cv::createMergeRobertson()
    //

    /**
     * Creates MergeRobertson object
     * @return automatically generated
     */
    public static MergeRobertson createMergeRobertson() {
        return MergeRobertson.__fromPtr__(createMergeRobertson_0());
    }


    //
    // C++:  Ptr_Tonemap cv::createTonemap(float gamma = 1.0f)
    //

    /**
     * Creates simple linear mapper with gamma correction
     *
     * @param gamma positive value for gamma correction. Gamma value of 1.0 implies no correction, gamma
     * equal to 2.2f is suitable for most displays.
     * Generally gamma &gt; 1 brightens the image and gamma &lt; 1 darkens it.
     * @return automatically generated
     */
    public static Tonemap createTonemap(float gamma) {
        return Tonemap.__fromPtr__(createTonemap_0(gamma));
    }

    /**
     * Creates simple linear mapper with gamma correction
     *
     * equal to 2.2f is suitable for most displays.
     * Generally gamma &gt; 1 brightens the image and gamma &lt; 1 darkens it.
     * @return automatically generated
     */
    public static Tonemap createTonemap() {
        return Tonemap.__fromPtr__(createTonemap_1());
    }


    //
    // C++:  Ptr_TonemapDrago cv::createTonemapDrago(float gamma = 1.0f, float saturation = 1.0f, float bias = 0.85f)
    //

    /**
     * Creates TonemapDrago object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * @param saturation positive saturation enhancement value. 1.0 preserves saturation, values greater
     * than 1 increase saturation and values less than 1 decrease it.
     * @param bias value for bias function in [0, 1] range. Values from 0.7 to 0.9 usually give best
     * results, default value is 0.85.
     * @return automatically generated
     */
    public static TonemapDrago createTonemapDrago(float gamma, float saturation, float bias) {
        return TonemapDrago.__fromPtr__(createTonemapDrago_0(gamma, saturation, bias));
    }

    /**
     * Creates TonemapDrago object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * @param saturation positive saturation enhancement value. 1.0 preserves saturation, values greater
     * than 1 increase saturation and values less than 1 decrease it.
     * results, default value is 0.85.
     * @return automatically generated
     */
    public static TonemapDrago createTonemapDrago(float gamma, float saturation) {
        return TonemapDrago.__fromPtr__(createTonemapDrago_1(gamma, saturation));
    }

    /**
     * Creates TonemapDrago object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * than 1 increase saturation and values less than 1 decrease it.
     * results, default value is 0.85.
     * @return automatically generated
     */
    public static TonemapDrago createTonemapDrago(float gamma) {
        return TonemapDrago.__fromPtr__(createTonemapDrago_2(gamma));
    }

    /**
     * Creates TonemapDrago object
     *
     * than 1 increase saturation and values less than 1 decrease it.
     * results, default value is 0.85.
     * @return automatically generated
     */
    public static TonemapDrago createTonemapDrago() {
        return TonemapDrago.__fromPtr__(createTonemapDrago_3());
    }


    //
    // C++:  Ptr_TonemapMantiuk cv::createTonemapMantiuk(float gamma = 1.0f, float scale = 0.7f, float saturation = 1.0f)
    //

    /**
     * Creates TonemapMantiuk object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * @param scale contrast scale factor. HVS response is multiplied by this parameter, thus compressing
     * dynamic range. Values from 0.6 to 0.9 produce best results.
     * @param saturation saturation enhancement value. See createTonemapDrago
     * @return automatically generated
     */
    public static TonemapMantiuk createTonemapMantiuk(float gamma, float scale, float saturation) {
        return TonemapMantiuk.__fromPtr__(createTonemapMantiuk_0(gamma, scale, saturation));
    }

    /**
     * Creates TonemapMantiuk object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * @param scale contrast scale factor. HVS response is multiplied by this parameter, thus compressing
     * dynamic range. Values from 0.6 to 0.9 produce best results.
     * @return automatically generated
     */
    public static TonemapMantiuk createTonemapMantiuk(float gamma, float scale) {
        return TonemapMantiuk.__fromPtr__(createTonemapMantiuk_1(gamma, scale));
    }

    /**
     * Creates TonemapMantiuk object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * dynamic range. Values from 0.6 to 0.9 produce best results.
     * @return automatically generated
     */
    public static TonemapMantiuk createTonemapMantiuk(float gamma) {
        return TonemapMantiuk.__fromPtr__(createTonemapMantiuk_2(gamma));
    }

    /**
     * Creates TonemapMantiuk object
     *
     * dynamic range. Values from 0.6 to 0.9 produce best results.
     * @return automatically generated
     */
    public static TonemapMantiuk createTonemapMantiuk() {
        return TonemapMantiuk.__fromPtr__(createTonemapMantiuk_3());
    }


    //
    // C++:  Ptr_TonemapReinhard cv::createTonemapReinhard(float gamma = 1.0f, float intensity = 0.0f, float light_adapt = 1.0f, float color_adapt = 0.0f)
    //

    /**
     * Creates TonemapReinhard object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * @param intensity result intensity in [-8, 8] range. Greater intensity produces brighter results.
     * @param light_adapt light adaptation in [0, 1] range. If 1 adaptation is based only on pixel
     * value, if 0 it's global, otherwise it's a weighted mean of this two cases.
     * @param color_adapt chromatic adaptation in [0, 1] range. If 1 channels are treated independently,
     * if 0 adaptation level is the same for each channel.
     * @return automatically generated
     */
    public static TonemapReinhard createTonemapReinhard(float gamma, float intensity, float light_adapt, float color_adapt) {
        return TonemapReinhard.__fromPtr__(createTonemapReinhard_0(gamma, intensity, light_adapt, color_adapt));
    }

    /**
     * Creates TonemapReinhard object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * @param intensity result intensity in [-8, 8] range. Greater intensity produces brighter results.
     * @param light_adapt light adaptation in [0, 1] range. If 1 adaptation is based only on pixel
     * value, if 0 it's global, otherwise it's a weighted mean of this two cases.
     * if 0 adaptation level is the same for each channel.
     * @return automatically generated
     */
    public static TonemapReinhard createTonemapReinhard(float gamma, float intensity, float light_adapt) {
        return TonemapReinhard.__fromPtr__(createTonemapReinhard_1(gamma, intensity, light_adapt));
    }

    /**
     * Creates TonemapReinhard object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * @param intensity result intensity in [-8, 8] range. Greater intensity produces brighter results.
     * value, if 0 it's global, otherwise it's a weighted mean of this two cases.
     * if 0 adaptation level is the same for each channel.
     * @return automatically generated
     */
    public static TonemapReinhard createTonemapReinhard(float gamma, float intensity) {
        return TonemapReinhard.__fromPtr__(createTonemapReinhard_2(gamma, intensity));
    }

    /**
     * Creates TonemapReinhard object
     *
     * @param gamma gamma value for gamma correction. See createTonemap
     * value, if 0 it's global, otherwise it's a weighted mean of this two cases.
     * if 0 adaptation level is the same for each channel.
     * @return automatically generated
     */
    public static TonemapReinhard createTonemapReinhard(float gamma) {
        return TonemapReinhard.__fromPtr__(createTonemapReinhard_3(gamma));
    }

    /**
     * Creates TonemapReinhard object
     *
     * value, if 0 it's global, otherwise it's a weighted mean of this two cases.
     * if 0 adaptation level is the same for each channel.
     * @return automatically generated
     */
    public static TonemapReinhard createTonemapReinhard() {
        return TonemapReinhard.__fromPtr__(createTonemapReinhard_4());
    }


    //
    // C++:  void cv::colorChange(Mat src, Mat mask, Mat& dst, float red_mul = 1.0f, float green_mul = 1.0f, float blue_mul = 1.0f)
    //

    /**
     * Given an original color image, two differently colored versions of this image can be mixed
     * seamlessly.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src .
     * @param red_mul R-channel multiply factor.
     * @param green_mul G-channel multiply factor.
     * @param blue_mul B-channel multiply factor.
     *
     * Multiplication factor is between .5 to 2.5.
     */
    public static void colorChange(Mat src, Mat mask, Mat dst, float red_mul, float green_mul, float blue_mul) {
        colorChange_0(src.nativeObj, mask.nativeObj, dst.nativeObj, red_mul, green_mul, blue_mul);
    }

    /**
     * Given an original color image, two differently colored versions of this image can be mixed
     * seamlessly.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src .
     * @param red_mul R-channel multiply factor.
     * @param green_mul G-channel multiply factor.
     *
     * Multiplication factor is between .5 to 2.5.
     */
    public static void colorChange(Mat src, Mat mask, Mat dst, float red_mul, float green_mul) {
        colorChange_1(src.nativeObj, mask.nativeObj, dst.nativeObj, red_mul, green_mul);
    }

    /**
     * Given an original color image, two differently colored versions of this image can be mixed
     * seamlessly.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src .
     * @param red_mul R-channel multiply factor.
     *
     * Multiplication factor is between .5 to 2.5.
     */
    public static void colorChange(Mat src, Mat mask, Mat dst, float red_mul) {
        colorChange_2(src.nativeObj, mask.nativeObj, dst.nativeObj, red_mul);
    }

    /**
     * Given an original color image, two differently colored versions of this image can be mixed
     * seamlessly.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src .
     *
     * Multiplication factor is between .5 to 2.5.
     */
    public static void colorChange(Mat src, Mat mask, Mat dst) {
        colorChange_3(src.nativeObj, mask.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::decolor(Mat src, Mat& grayscale, Mat& color_boost)
    //

    /**
     * Transforms a color image to a grayscale image. It is a basic tool in digital printing, stylized
     * black-and-white photograph rendering, and in many single channel image processing applications
     * CITE: CL12 .
     *
     * @param src Input 8-bit 3-channel image.
     * @param grayscale Output 8-bit 1-channel image.
     * @param color_boost Output 8-bit 3-channel image.
     *
     * This function is to be applied on color images.
     */
    public static void decolor(Mat src, Mat grayscale, Mat color_boost) {
        decolor_0(src.nativeObj, grayscale.nativeObj, color_boost.nativeObj);
    }


    //
    // C++:  void cv::denoise_TVL1(vector_Mat observations, Mat result, double lambda = 1.0, int niters = 30)
    //

    /**
     * Primal-dual algorithm is an algorithm for solving special types of variational problems (that is,
     * finding a function to minimize some functional). As the image denoising, in particular, may be seen
     * as the variational problem, primal-dual algorithm then can be used to perform denoising and this is
     * exactly what is implemented.
     *
     * It should be noted, that this implementation was taken from the July 2013 blog entry
     * CITE: MA13 , which also contained (slightly more general) ready-to-use source code on Python.
     * Subsequently, that code was rewritten on C++ with the usage of openCV by Vadim Pisarevsky at the end
     * of July 2013 and finally it was slightly adapted by later authors.
     *
     * Although the thorough discussion and justification of the algorithm involved may be found in
     * CITE: ChambolleEtAl, it might make sense to skim over it here, following CITE: MA13 . To begin
     * with, we consider the 1-byte gray-level images as the functions from the rectangular domain of
     * pixels (it may be seen as set
     * \(\left\{(x,y)\in\mathbb{N}\times\mathbb{N}\mid 1\leq x\leq n,\;1\leq y\leq m\right\}\) for some
     * \(m,\;n\in\mathbb{N}\)) into \(\{0,1,\dots,255\}\). We shall denote the noised images as \(f_i\) and with
     * this view, given some image \(x\) of the same size, we may measure how bad it is by the formula
     *
     * \(\left\|\left\|\nabla x\right\|\right\| + \lambda\sum_i\left\|\left\|x-f_i\right\|\right\|\)
     *
     * \(\|\|\cdot\|\|\) here denotes \(L_2\)-norm and as you see, the first addend states that we want our
     * image to be smooth (ideally, having zero gradient, thus being constant) and the second states that
     * we want our result to be close to the observations we've got. If we treat \(x\) as a function, this is
     * exactly the functional what we seek to minimize and here the Primal-Dual algorithm comes into play.
     *
     * @param observations This array should contain one or more noised versions of the image that is to
     * be restored.
     * @param result Here the denoised image will be stored. There is no need to do pre-allocation of
     * storage space, as it will be automatically allocated, if necessary.
     * @param lambda Corresponds to \(\lambda\) in the formulas above. As it is enlarged, the smooth
     * (blurred) images are treated more favorably than detailed (but maybe more noised) ones. Roughly
     * speaking, as it becomes smaller, the result will be more blur but more sever outliers will be
     * removed.
     * @param niters Number of iterations that the algorithm will run. Of course, as more iterations as
     * better, but it is hard to quantitatively refine this statement, so just use the default and
     * increase it if the results are poor.
     */
    public static void denoise_TVL1(List<Mat> observations, Mat result, double lambda, int niters) {
        Mat observations_mat = Converters.vector_Mat_to_Mat(observations);
        denoise_TVL1_0(observations_mat.nativeObj, result.nativeObj, lambda, niters);
    }

    /**
     * Primal-dual algorithm is an algorithm for solving special types of variational problems (that is,
     * finding a function to minimize some functional). As the image denoising, in particular, may be seen
     * as the variational problem, primal-dual algorithm then can be used to perform denoising and this is
     * exactly what is implemented.
     *
     * It should be noted, that this implementation was taken from the July 2013 blog entry
     * CITE: MA13 , which also contained (slightly more general) ready-to-use source code on Python.
     * Subsequently, that code was rewritten on C++ with the usage of openCV by Vadim Pisarevsky at the end
     * of July 2013 and finally it was slightly adapted by later authors.
     *
     * Although the thorough discussion and justification of the algorithm involved may be found in
     * CITE: ChambolleEtAl, it might make sense to skim over it here, following CITE: MA13 . To begin
     * with, we consider the 1-byte gray-level images as the functions from the rectangular domain of
     * pixels (it may be seen as set
     * \(\left\{(x,y)\in\mathbb{N}\times\mathbb{N}\mid 1\leq x\leq n,\;1\leq y\leq m\right\}\) for some
     * \(m,\;n\in\mathbb{N}\)) into \(\{0,1,\dots,255\}\). We shall denote the noised images as \(f_i\) and with
     * this view, given some image \(x\) of the same size, we may measure how bad it is by the formula
     *
     * \(\left\|\left\|\nabla x\right\|\right\| + \lambda\sum_i\left\|\left\|x-f_i\right\|\right\|\)
     *
     * \(\|\|\cdot\|\|\) here denotes \(L_2\)-norm and as you see, the first addend states that we want our
     * image to be smooth (ideally, having zero gradient, thus being constant) and the second states that
     * we want our result to be close to the observations we've got. If we treat \(x\) as a function, this is
     * exactly the functional what we seek to minimize and here the Primal-Dual algorithm comes into play.
     *
     * @param observations This array should contain one or more noised versions of the image that is to
     * be restored.
     * @param result Here the denoised image will be stored. There is no need to do pre-allocation of
     * storage space, as it will be automatically allocated, if necessary.
     * @param lambda Corresponds to \(\lambda\) in the formulas above. As it is enlarged, the smooth
     * (blurred) images are treated more favorably than detailed (but maybe more noised) ones. Roughly
     * speaking, as it becomes smaller, the result will be more blur but more sever outliers will be
     * removed.
     * better, but it is hard to quantitatively refine this statement, so just use the default and
     * increase it if the results are poor.
     */
    public static void denoise_TVL1(List<Mat> observations, Mat result, double lambda) {
        Mat observations_mat = Converters.vector_Mat_to_Mat(observations);
        denoise_TVL1_1(observations_mat.nativeObj, result.nativeObj, lambda);
    }

    /**
     * Primal-dual algorithm is an algorithm for solving special types of variational problems (that is,
     * finding a function to minimize some functional). As the image denoising, in particular, may be seen
     * as the variational problem, primal-dual algorithm then can be used to perform denoising and this is
     * exactly what is implemented.
     *
     * It should be noted, that this implementation was taken from the July 2013 blog entry
     * CITE: MA13 , which also contained (slightly more general) ready-to-use source code on Python.
     * Subsequently, that code was rewritten on C++ with the usage of openCV by Vadim Pisarevsky at the end
     * of July 2013 and finally it was slightly adapted by later authors.
     *
     * Although the thorough discussion and justification of the algorithm involved may be found in
     * CITE: ChambolleEtAl, it might make sense to skim over it here, following CITE: MA13 . To begin
     * with, we consider the 1-byte gray-level images as the functions from the rectangular domain of
     * pixels (it may be seen as set
     * \(\left\{(x,y)\in\mathbb{N}\times\mathbb{N}\mid 1\leq x\leq n,\;1\leq y\leq m\right\}\) for some
     * \(m,\;n\in\mathbb{N}\)) into \(\{0,1,\dots,255\}\). We shall denote the noised images as \(f_i\) and with
     * this view, given some image \(x\) of the same size, we may measure how bad it is by the formula
     *
     * \(\left\|\left\|\nabla x\right\|\right\| + \lambda\sum_i\left\|\left\|x-f_i\right\|\right\|\)
     *
     * \(\|\|\cdot\|\|\) here denotes \(L_2\)-norm and as you see, the first addend states that we want our
     * image to be smooth (ideally, having zero gradient, thus being constant) and the second states that
     * we want our result to be close to the observations we've got. If we treat \(x\) as a function, this is
     * exactly the functional what we seek to minimize and here the Primal-Dual algorithm comes into play.
     *
     * @param observations This array should contain one or more noised versions of the image that is to
     * be restored.
     * @param result Here the denoised image will be stored. There is no need to do pre-allocation of
     * storage space, as it will be automatically allocated, if necessary.
     * (blurred) images are treated more favorably than detailed (but maybe more noised) ones. Roughly
     * speaking, as it becomes smaller, the result will be more blur but more sever outliers will be
     * removed.
     * better, but it is hard to quantitatively refine this statement, so just use the default and
     * increase it if the results are poor.
     */
    public static void denoise_TVL1(List<Mat> observations, Mat result) {
        Mat observations_mat = Converters.vector_Mat_to_Mat(observations);
        denoise_TVL1_2(observations_mat.nativeObj, result.nativeObj);
    }


    //
    // C++:  void cv::detailEnhance(Mat src, Mat& dst, float sigma_s = 10, float sigma_r = 0.15f)
    //

    /**
     * This filter enhances the details of a particular image.
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param sigma_s %Range between 0 to 200.
     * @param sigma_r %Range between 0 to 1.
     */
    public static void detailEnhance(Mat src, Mat dst, float sigma_s, float sigma_r) {
        detailEnhance_0(src.nativeObj, dst.nativeObj, sigma_s, sigma_r);
    }

    /**
     * This filter enhances the details of a particular image.
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param sigma_s %Range between 0 to 200.
     */
    public static void detailEnhance(Mat src, Mat dst, float sigma_s) {
        detailEnhance_1(src.nativeObj, dst.nativeObj, sigma_s);
    }

    /**
     * This filter enhances the details of a particular image.
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src.
     */
    public static void detailEnhance(Mat src, Mat dst) {
        detailEnhance_2(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::edgePreservingFilter(Mat src, Mat& dst, int flags = 1, float sigma_s = 60, float sigma_r = 0.4f)
    //

    /**
     * Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing
     * filters are used in many different applications CITE: EM11 .
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output 8-bit 3-channel image.
     * @param flags Edge preserving filters: cv::RECURS_FILTER or cv::NORMCONV_FILTER
     * @param sigma_s %Range between 0 to 200.
     * @param sigma_r %Range between 0 to 1.
     */
    public static void edgePreservingFilter(Mat src, Mat dst, int flags, float sigma_s, float sigma_r) {
        edgePreservingFilter_0(src.nativeObj, dst.nativeObj, flags, sigma_s, sigma_r);
    }

    /**
     * Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing
     * filters are used in many different applications CITE: EM11 .
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output 8-bit 3-channel image.
     * @param flags Edge preserving filters: cv::RECURS_FILTER or cv::NORMCONV_FILTER
     * @param sigma_s %Range between 0 to 200.
     */
    public static void edgePreservingFilter(Mat src, Mat dst, int flags, float sigma_s) {
        edgePreservingFilter_1(src.nativeObj, dst.nativeObj, flags, sigma_s);
    }

    /**
     * Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing
     * filters are used in many different applications CITE: EM11 .
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output 8-bit 3-channel image.
     * @param flags Edge preserving filters: cv::RECURS_FILTER or cv::NORMCONV_FILTER
     */
    public static void edgePreservingFilter(Mat src, Mat dst, int flags) {
        edgePreservingFilter_2(src.nativeObj, dst.nativeObj, flags);
    }

    /**
     * Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing
     * filters are used in many different applications CITE: EM11 .
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output 8-bit 3-channel image.
     */
    public static void edgePreservingFilter(Mat src, Mat dst) {
        edgePreservingFilter_3(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::fastNlMeansDenoising(Mat src, Mat& dst, float h = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    //

    /**
     * Perform image denoising using Non-local Means Denoising algorithm
     * &lt;http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/&gt; with several computational
     * optimizations. Noise expected to be a gaussian white noise
     *
     * @param src Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.
     * @param dst Output image with the same size and type as src .
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * @param searchWindowSize Size in pixels of the window that is used to compute weighted average for
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength. Big h value perfectly removes noise but also
     * removes image details, smaller h value preserves details but also preserves some noise
     *
     * This function expected to be applied to grayscale images. For colored images look at
     * fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
     * image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
     * image to CIELAB colorspace and then separately denoise L and AB components with different h
     * parameter.
     */
    public static void fastNlMeansDenoising(Mat src, Mat dst, float h, int templateWindowSize, int searchWindowSize) {
        fastNlMeansDenoising_0(src.nativeObj, dst.nativeObj, h, templateWindowSize, searchWindowSize);
    }

    /**
     * Perform image denoising using Non-local Means Denoising algorithm
     * &lt;http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/&gt; with several computational
     * optimizations. Noise expected to be a gaussian white noise
     *
     * @param src Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.
     * @param dst Output image with the same size and type as src .
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength. Big h value perfectly removes noise but also
     * removes image details, smaller h value preserves details but also preserves some noise
     *
     * This function expected to be applied to grayscale images. For colored images look at
     * fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
     * image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
     * image to CIELAB colorspace and then separately denoise L and AB components with different h
     * parameter.
     */
    public static void fastNlMeansDenoising(Mat src, Mat dst, float h, int templateWindowSize) {
        fastNlMeansDenoising_1(src.nativeObj, dst.nativeObj, h, templateWindowSize);
    }

    /**
     * Perform image denoising using Non-local Means Denoising algorithm
     * &lt;http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/&gt; with several computational
     * optimizations. Noise expected to be a gaussian white noise
     *
     * @param src Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.
     * @param dst Output image with the same size and type as src .
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength. Big h value perfectly removes noise but also
     * removes image details, smaller h value preserves details but also preserves some noise
     *
     * This function expected to be applied to grayscale images. For colored images look at
     * fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
     * image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
     * image to CIELAB colorspace and then separately denoise L and AB components with different h
     * parameter.
     */
    public static void fastNlMeansDenoising(Mat src, Mat dst, float h) {
        fastNlMeansDenoising_2(src.nativeObj, dst.nativeObj, h);
    }

    /**
     * Perform image denoising using Non-local Means Denoising algorithm
     * &lt;http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/&gt; with several computational
     * optimizations. Noise expected to be a gaussian white noise
     *
     * @param src Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.
     * @param dst Output image with the same size and type as src .
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * removes image details, smaller h value preserves details but also preserves some noise
     *
     * This function expected to be applied to grayscale images. For colored images look at
     * fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
     * image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
     * image to CIELAB colorspace and then separately denoise L and AB components with different h
     * parameter.
     */
    public static void fastNlMeansDenoising(Mat src, Mat dst) {
        fastNlMeansDenoising_3(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::fastNlMeansDenoising(Mat src, Mat& dst, vector_float h, int templateWindowSize = 7, int searchWindowSize = 21, int normType = NORM_L2)
    //

    /**
     * Perform image denoising using Non-local Means Denoising algorithm
     * &lt;http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/&gt; with several computational
     * optimizations. Noise expected to be a gaussian white noise
     *
     * @param src Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
     * 2-channel, 3-channel or 4-channel image.
     * @param dst Output image with the same size and type as src .
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * @param searchWindowSize Size in pixels of the window that is used to compute weighted average for
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Array of parameters regulating filter strength, either one
     * parameter applied to all channels or one per channel in dst. Big h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     * @param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1
     *
     * This function expected to be applied to grayscale images. For colored images look at
     * fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
     * image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
     * image to CIELAB colorspace and then separately denoise L and AB components with different h
     * parameter.
     */
    public static void fastNlMeansDenoising(Mat src, Mat dst, MatOfFloat h, int templateWindowSize, int searchWindowSize, int normType) {
        Mat h_mat = h;
        fastNlMeansDenoising_4(src.nativeObj, dst.nativeObj, h_mat.nativeObj, templateWindowSize, searchWindowSize, normType);
    }

    /**
     * Perform image denoising using Non-local Means Denoising algorithm
     * &lt;http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/&gt; with several computational
     * optimizations. Noise expected to be a gaussian white noise
     *
     * @param src Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
     * 2-channel, 3-channel or 4-channel image.
     * @param dst Output image with the same size and type as src .
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * @param searchWindowSize Size in pixels of the window that is used to compute weighted average for
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Array of parameters regulating filter strength, either one
     * parameter applied to all channels or one per channel in dst. Big h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     *
     * This function expected to be applied to grayscale images. For colored images look at
     * fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
     * image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
     * image to CIELAB colorspace and then separately denoise L and AB components with different h
     * parameter.
     */
    public static void fastNlMeansDenoising(Mat src, Mat dst, MatOfFloat h, int templateWindowSize, int searchWindowSize) {
        Mat h_mat = h;
        fastNlMeansDenoising_5(src.nativeObj, dst.nativeObj, h_mat.nativeObj, templateWindowSize, searchWindowSize);
    }

    /**
     * Perform image denoising using Non-local Means Denoising algorithm
     * &lt;http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/&gt; with several computational
     * optimizations. Noise expected to be a gaussian white noise
     *
     * @param src Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
     * 2-channel, 3-channel or 4-channel image.
     * @param dst Output image with the same size and type as src .
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Array of parameters regulating filter strength, either one
     * parameter applied to all channels or one per channel in dst. Big h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     *
     * This function expected to be applied to grayscale images. For colored images look at
     * fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
     * image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
     * image to CIELAB colorspace and then separately denoise L and AB components with different h
     * parameter.
     */
    public static void fastNlMeansDenoising(Mat src, Mat dst, MatOfFloat h, int templateWindowSize) {
        Mat h_mat = h;
        fastNlMeansDenoising_6(src.nativeObj, dst.nativeObj, h_mat.nativeObj, templateWindowSize);
    }

    /**
     * Perform image denoising using Non-local Means Denoising algorithm
     * &lt;http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/&gt; with several computational
     * optimizations. Noise expected to be a gaussian white noise
     *
     * @param src Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
     * 2-channel, 3-channel or 4-channel image.
     * @param dst Output image with the same size and type as src .
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Array of parameters regulating filter strength, either one
     * parameter applied to all channels or one per channel in dst. Big h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     *
     * This function expected to be applied to grayscale images. For colored images look at
     * fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
     * image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
     * image to CIELAB colorspace and then separately denoise L and AB components with different h
     * parameter.
     */
    public static void fastNlMeansDenoising(Mat src, Mat dst, MatOfFloat h) {
        Mat h_mat = h;
        fastNlMeansDenoising_7(src.nativeObj, dst.nativeObj, h_mat.nativeObj);
    }


    //
    // C++:  void cv::fastNlMeansDenoisingColored(Mat src, Mat& dst, float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    //

    /**
     * Modification of fastNlMeansDenoising function for colored images
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src .
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * @param searchWindowSize Size in pixels of the window that is used to compute weighted average for
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise
     * @param hColor The same as h but for color components. For most images value equals 10
     * will be enough to remove colored noise and do not distort colors
     *
     * The function converts image to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoising function.
     */
    public static void fastNlMeansDenoisingColored(Mat src, Mat dst, float h, float hColor, int templateWindowSize, int searchWindowSize) {
        fastNlMeansDenoisingColored_0(src.nativeObj, dst.nativeObj, h, hColor, templateWindowSize, searchWindowSize);
    }

    /**
     * Modification of fastNlMeansDenoising function for colored images
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src .
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise
     * @param hColor The same as h but for color components. For most images value equals 10
     * will be enough to remove colored noise and do not distort colors
     *
     * The function converts image to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoising function.
     */
    public static void fastNlMeansDenoisingColored(Mat src, Mat dst, float h, float hColor, int templateWindowSize) {
        fastNlMeansDenoisingColored_1(src.nativeObj, dst.nativeObj, h, hColor, templateWindowSize);
    }

    /**
     * Modification of fastNlMeansDenoising function for colored images
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src .
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise
     * @param hColor The same as h but for color components. For most images value equals 10
     * will be enough to remove colored noise and do not distort colors
     *
     * The function converts image to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoising function.
     */
    public static void fastNlMeansDenoisingColored(Mat src, Mat dst, float h, float hColor) {
        fastNlMeansDenoisingColored_2(src.nativeObj, dst.nativeObj, h, hColor);
    }

    /**
     * Modification of fastNlMeansDenoising function for colored images
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src .
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise
     * will be enough to remove colored noise and do not distort colors
     *
     * The function converts image to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoising function.
     */
    public static void fastNlMeansDenoisingColored(Mat src, Mat dst, float h) {
        fastNlMeansDenoisingColored_3(src.nativeObj, dst.nativeObj, h);
    }

    /**
     * Modification of fastNlMeansDenoising function for colored images
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src .
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise
     * will be enough to remove colored noise and do not distort colors
     *
     * The function converts image to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoising function.
     */
    public static void fastNlMeansDenoisingColored(Mat src, Mat dst) {
        fastNlMeansDenoisingColored_4(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::fastNlMeansDenoisingColoredMulti(vector_Mat srcImgs, Mat& dst, int imgToDenoiseIndex, int temporalWindowSize, float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    //

    /**
     * Modification of fastNlMeansDenoisingMulti function for colored images sequences
     *
     * @param srcImgs Input 8-bit 3-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * @param searchWindowSize Size in pixels of the window that is used to compute weighted average for
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise.
     * @param hColor The same as h but for color components.
     *
     * The function converts images to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoisingMulti function.
     */
    public static void fastNlMeansDenoisingColoredMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, float h, float hColor, int templateWindowSize, int searchWindowSize) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingColoredMulti_0(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h, hColor, templateWindowSize, searchWindowSize);
    }

    /**
     * Modification of fastNlMeansDenoisingMulti function for colored images sequences
     *
     * @param srcImgs Input 8-bit 3-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise.
     * @param hColor The same as h but for color components.
     *
     * The function converts images to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoisingMulti function.
     */
    public static void fastNlMeansDenoisingColoredMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, float h, float hColor, int templateWindowSize) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingColoredMulti_1(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h, hColor, templateWindowSize);
    }

    /**
     * Modification of fastNlMeansDenoisingMulti function for colored images sequences
     *
     * @param srcImgs Input 8-bit 3-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise.
     * @param hColor The same as h but for color components.
     *
     * The function converts images to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoisingMulti function.
     */
    public static void fastNlMeansDenoisingColoredMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, float h, float hColor) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingColoredMulti_2(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h, hColor);
    }

    /**
     * Modification of fastNlMeansDenoisingMulti function for colored images sequences
     *
     * @param srcImgs Input 8-bit 3-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise.
     *
     * The function converts images to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoisingMulti function.
     */
    public static void fastNlMeansDenoisingColoredMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, float h) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingColoredMulti_3(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h);
    }

    /**
     * Modification of fastNlMeansDenoisingMulti function for colored images sequences
     *
     * @param srcImgs Input 8-bit 3-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * removes noise but also removes image details, smaller h value preserves details but also preserves
     * some noise.
     *
     * The function converts images to CIELAB colorspace and then separately denoise L and AB components
     * with given h parameters using fastNlMeansDenoisingMulti function.
     */
    public static void fastNlMeansDenoisingColoredMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingColoredMulti_4(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize);
    }


    //
    // C++:  void cv::fastNlMeansDenoisingMulti(vector_Mat srcImgs, Mat& dst, int imgToDenoiseIndex, int temporalWindowSize, float h = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    //

    /**
     * Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
     * captured in small period of time. For example video. This version of the function is for grayscale
     * images or for manual manipulation with colorspaces. For more details see
     * &lt;http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394&gt;
     *
     * @param srcImgs Input 8-bit 1-channel, 2-channel, 3-channel or
     * 4-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * @param searchWindowSize Size in pixels of the window that is used to compute weighted average for
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength. Bigger h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     */
    public static void fastNlMeansDenoisingMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, float h, int templateWindowSize, int searchWindowSize) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingMulti_0(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize);
    }

    /**
     * Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
     * captured in small period of time. For example video. This version of the function is for grayscale
     * images or for manual manipulation with colorspaces. For more details see
     * &lt;http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394&gt;
     *
     * @param srcImgs Input 8-bit 1-channel, 2-channel, 3-channel or
     * 4-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength. Bigger h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     */
    public static void fastNlMeansDenoisingMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, float h, int templateWindowSize) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingMulti_1(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize);
    }

    /**
     * Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
     * captured in small period of time. For example video. This version of the function is for grayscale
     * images or for manual manipulation with colorspaces. For more details see
     * &lt;http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394&gt;
     *
     * @param srcImgs Input 8-bit 1-channel, 2-channel, 3-channel or
     * 4-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Parameter regulating filter strength. Bigger h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     */
    public static void fastNlMeansDenoisingMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, float h) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingMulti_2(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h);
    }

    /**
     * Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
     * captured in small period of time. For example video. This version of the function is for grayscale
     * images or for manual manipulation with colorspaces. For more details see
     * &lt;http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394&gt;
     *
     * @param srcImgs Input 8-bit 1-channel, 2-channel, 3-channel or
     * 4-channel images sequence. All images should have the same type and
     * size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     */
    public static void fastNlMeansDenoisingMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        fastNlMeansDenoisingMulti_3(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize);
    }


    //
    // C++:  void cv::fastNlMeansDenoisingMulti(vector_Mat srcImgs, Mat& dst, int imgToDenoiseIndex, int temporalWindowSize, vector_float h, int templateWindowSize = 7, int searchWindowSize = 21, int normType = NORM_L2)
    //

    /**
     * Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
     * captured in small period of time. For example video. This version of the function is for grayscale
     * images or for manual manipulation with colorspaces. For more details see
     * &lt;http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394&gt;
     *
     * @param srcImgs Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
     * 2-channel, 3-channel or 4-channel images sequence. All images should
     * have the same type and size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * @param searchWindowSize Size in pixels of the window that is used to compute weighted average for
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Array of parameters regulating filter strength, either one
     * parameter applied to all channels or one per channel in dst. Big h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     * @param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1
     */
    public static void fastNlMeansDenoisingMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, MatOfFloat h, int templateWindowSize, int searchWindowSize, int normType) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        Mat h_mat = h;
        fastNlMeansDenoisingMulti_4(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h_mat.nativeObj, templateWindowSize, searchWindowSize, normType);
    }

    /**
     * Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
     * captured in small period of time. For example video. This version of the function is for grayscale
     * images or for manual manipulation with colorspaces. For more details see
     * &lt;http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394&gt;
     *
     * @param srcImgs Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
     * 2-channel, 3-channel or 4-channel images sequence. All images should
     * have the same type and size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * @param searchWindowSize Size in pixels of the window that is used to compute weighted average for
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Array of parameters regulating filter strength, either one
     * parameter applied to all channels or one per channel in dst. Big h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     */
    public static void fastNlMeansDenoisingMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, MatOfFloat h, int templateWindowSize, int searchWindowSize) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        Mat h_mat = h;
        fastNlMeansDenoisingMulti_5(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h_mat.nativeObj, templateWindowSize, searchWindowSize);
    }

    /**
     * Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
     * captured in small period of time. For example video. This version of the function is for grayscale
     * images or for manual manipulation with colorspaces. For more details see
     * &lt;http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394&gt;
     *
     * @param srcImgs Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
     * 2-channel, 3-channel or 4-channel images sequence. All images should
     * have the same type and size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * @param templateWindowSize Size in pixels of the template patch that is used to compute weights.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Array of parameters regulating filter strength, either one
     * parameter applied to all channels or one per channel in dst. Big h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     */
    public static void fastNlMeansDenoisingMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, MatOfFloat h, int templateWindowSize) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        Mat h_mat = h;
        fastNlMeansDenoisingMulti_6(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h_mat.nativeObj, templateWindowSize);
    }

    /**
     * Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
     * captured in small period of time. For example video. This version of the function is for grayscale
     * images or for manual manipulation with colorspaces. For more details see
     * &lt;http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394&gt;
     *
     * @param srcImgs Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
     * 2-channel, 3-channel or 4-channel images sequence. All images should
     * have the same type and size.
     * @param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
     * @param temporalWindowSize Number of surrounding images to use for target image denoising. Should
     * be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
     * imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
     * srcImgs[imgToDenoiseIndex] image.
     * @param dst Output image with the same size and type as srcImgs images.
     * Should be odd. Recommended value 7 pixels
     * given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
     * denoising time. Recommended value 21 pixels
     * @param h Array of parameters regulating filter strength, either one
     * parameter applied to all channels or one per channel in dst. Big h value
     * perfectly removes noise but also removes image details, smaller h
     * value preserves details but also preserves some noise
     */
    public static void fastNlMeansDenoisingMulti(List<Mat> srcImgs, Mat dst, int imgToDenoiseIndex, int temporalWindowSize, MatOfFloat h) {
        Mat srcImgs_mat = Converters.vector_Mat_to_Mat(srcImgs);
        Mat h_mat = h;
        fastNlMeansDenoisingMulti_7(srcImgs_mat.nativeObj, dst.nativeObj, imgToDenoiseIndex, temporalWindowSize, h_mat.nativeObj);
    }


    //
    // C++:  void cv::illuminationChange(Mat src, Mat mask, Mat& dst, float alpha = 0.2f, float beta = 0.4f)
    //

    /**
     * Applying an appropriate non-linear transformation to the gradient field inside the selection and
     * then integrating back with a Poisson solver, modifies locally the apparent illumination of an image.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param alpha Value ranges between 0-2.
     * @param beta Value ranges between 0-2.
     *
     * This is useful to highlight under-exposed foreground objects or to reduce specular reflections.
     */
    public static void illuminationChange(Mat src, Mat mask, Mat dst, float alpha, float beta) {
        illuminationChange_0(src.nativeObj, mask.nativeObj, dst.nativeObj, alpha, beta);
    }

    /**
     * Applying an appropriate non-linear transformation to the gradient field inside the selection and
     * then integrating back with a Poisson solver, modifies locally the apparent illumination of an image.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param alpha Value ranges between 0-2.
     *
     * This is useful to highlight under-exposed foreground objects or to reduce specular reflections.
     */
    public static void illuminationChange(Mat src, Mat mask, Mat dst, float alpha) {
        illuminationChange_1(src.nativeObj, mask.nativeObj, dst.nativeObj, alpha);
    }

    /**
     * Applying an appropriate non-linear transformation to the gradient field inside the selection and
     * then integrating back with a Poisson solver, modifies locally the apparent illumination of an image.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src.
     *
     * This is useful to highlight under-exposed foreground objects or to reduce specular reflections.
     */
    public static void illuminationChange(Mat src, Mat mask, Mat dst) {
        illuminationChange_2(src.nativeObj, mask.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::inpaint(Mat src, Mat inpaintMask, Mat& dst, double inpaintRadius, int flags)
    //

    /**
     * Restores the selected region in an image using the region neighborhood.
     *
     * @param src Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.
     * @param inpaintMask Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that
     * needs to be inpainted.
     * @param dst Output image with the same size and type as src .
     * @param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered
     * by the algorithm.
     * @param flags Inpainting method that could be cv::INPAINT_NS or cv::INPAINT_TELEA
     *
     * The function reconstructs the selected image area from the pixel near the area boundary. The
     * function may be used to remove dust and scratches from a scanned photo, or to remove undesirable
     * objects from still images or video. See &lt;http://en.wikipedia.org/wiki/Inpainting&gt; for more details.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       An example using the inpainting technique can be found at
     *         opencv_source_code/samples/cpp/inpaint.cpp
     *   </li>
     *   <li>
     *       (Python) An example using the inpainting technique can be found at
     *         opencv_source_code/samples/python/inpaint.py
     *   </li>
     * </ul>
     */
    public static void inpaint(Mat src, Mat inpaintMask, Mat dst, double inpaintRadius, int flags) {
        inpaint_0(src.nativeObj, inpaintMask.nativeObj, dst.nativeObj, inpaintRadius, flags);
    }


    //
    // C++:  void cv::pencilSketch(Mat src, Mat& dst1, Mat& dst2, float sigma_s = 60, float sigma_r = 0.07f, float shade_factor = 0.02f)
    //

    /**
     * Pencil-like non-photorealistic line drawing
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst1 Output 8-bit 1-channel image.
     * @param dst2 Output image with the same size and type as src.
     * @param sigma_s %Range between 0 to 200.
     * @param sigma_r %Range between 0 to 1.
     * @param shade_factor %Range between 0 to 0.1.
     */
    public static void pencilSketch(Mat src, Mat dst1, Mat dst2, float sigma_s, float sigma_r, float shade_factor) {
        pencilSketch_0(src.nativeObj, dst1.nativeObj, dst2.nativeObj, sigma_s, sigma_r, shade_factor);
    }

    /**
     * Pencil-like non-photorealistic line drawing
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst1 Output 8-bit 1-channel image.
     * @param dst2 Output image with the same size and type as src.
     * @param sigma_s %Range between 0 to 200.
     * @param sigma_r %Range between 0 to 1.
     */
    public static void pencilSketch(Mat src, Mat dst1, Mat dst2, float sigma_s, float sigma_r) {
        pencilSketch_1(src.nativeObj, dst1.nativeObj, dst2.nativeObj, sigma_s, sigma_r);
    }

    /**
     * Pencil-like non-photorealistic line drawing
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst1 Output 8-bit 1-channel image.
     * @param dst2 Output image with the same size and type as src.
     * @param sigma_s %Range between 0 to 200.
     */
    public static void pencilSketch(Mat src, Mat dst1, Mat dst2, float sigma_s) {
        pencilSketch_2(src.nativeObj, dst1.nativeObj, dst2.nativeObj, sigma_s);
    }

    /**
     * Pencil-like non-photorealistic line drawing
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst1 Output 8-bit 1-channel image.
     * @param dst2 Output image with the same size and type as src.
     */
    public static void pencilSketch(Mat src, Mat dst1, Mat dst2) {
        pencilSketch_3(src.nativeObj, dst1.nativeObj, dst2.nativeObj);
    }


    //
    // C++:  void cv::seamlessClone(Mat src, Mat dst, Mat mask, Point p, Mat& blend, int flags)
    //

    /**
     * Image editing tasks concern either global changes (color/intensity corrections, filters,
     * deformations) or local changes concerned to a selection. Here we are interested in achieving local
     * changes, ones that are restricted to a region manually selected (ROI), in a seamless and effortless
     * manner. The extent of the changes ranges from slight distortions to complete replacement by novel
     * content CITE: PM03 .
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param p Point in dst image where object is placed.
     * @param blend Output image with the same size and type as dst.
     * @param flags Cloning method that could be cv::NORMAL_CLONE, cv::MIXED_CLONE or cv::MONOCHROME_TRANSFER
     */
    public static void seamlessClone(Mat src, Mat dst, Mat mask, Point p, Mat blend, int flags) {
        seamlessClone_0(src.nativeObj, dst.nativeObj, mask.nativeObj, p.x, p.y, blend.nativeObj, flags);
    }


    //
    // C++:  void cv::stylization(Mat src, Mat& dst, float sigma_s = 60, float sigma_r = 0.45f)
    //

    /**
     * Stylization aims to produce digital imagery with a wide variety of effects not focused on
     * photorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low
     * contrast while preserving, or enhancing, high-contrast features.
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param sigma_s %Range between 0 to 200.
     * @param sigma_r %Range between 0 to 1.
     */
    public static void stylization(Mat src, Mat dst, float sigma_s, float sigma_r) {
        stylization_0(src.nativeObj, dst.nativeObj, sigma_s, sigma_r);
    }

    /**
     * Stylization aims to produce digital imagery with a wide variety of effects not focused on
     * photorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low
     * contrast while preserving, or enhancing, high-contrast features.
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param sigma_s %Range between 0 to 200.
     */
    public static void stylization(Mat src, Mat dst, float sigma_s) {
        stylization_1(src.nativeObj, dst.nativeObj, sigma_s);
    }

    /**
     * Stylization aims to produce digital imagery with a wide variety of effects not focused on
     * photorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low
     * contrast while preserving, or enhancing, high-contrast features.
     *
     * @param src Input 8-bit 3-channel image.
     * @param dst Output image with the same size and type as src.
     */
    public static void stylization(Mat src, Mat dst) {
        stylization_2(src.nativeObj, dst.nativeObj);
    }


    //
    // C++:  void cv::textureFlattening(Mat src, Mat mask, Mat& dst, float low_threshold = 30, float high_threshold = 45, int kernel_size = 3)
    //

    /**
     * By retaining only the gradients at edge locations, before integrating with the Poisson solver, one
     * washes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge %Detector is used.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param low_threshold %Range from 0 to 100.
     * @param high_threshold Value &gt; 100.
     * @param kernel_size The size of the Sobel kernel to be used.
     *
     * <b>Note:</b>
     * The algorithm assumes that the color of the source image is close to that of the destination. This
     * assumption means that when the colors don't match, the source image color gets tinted toward the
     * color of the destination image.
     */
    public static void textureFlattening(Mat src, Mat mask, Mat dst, float low_threshold, float high_threshold, int kernel_size) {
        textureFlattening_0(src.nativeObj, mask.nativeObj, dst.nativeObj, low_threshold, high_threshold, kernel_size);
    }

    /**
     * By retaining only the gradients at edge locations, before integrating with the Poisson solver, one
     * washes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge %Detector is used.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param low_threshold %Range from 0 to 100.
     * @param high_threshold Value &gt; 100.
     *
     * <b>Note:</b>
     * The algorithm assumes that the color of the source image is close to that of the destination. This
     * assumption means that when the colors don't match, the source image color gets tinted toward the
     * color of the destination image.
     */
    public static void textureFlattening(Mat src, Mat mask, Mat dst, float low_threshold, float high_threshold) {
        textureFlattening_1(src.nativeObj, mask.nativeObj, dst.nativeObj, low_threshold, high_threshold);
    }

    /**
     * By retaining only the gradients at edge locations, before integrating with the Poisson solver, one
     * washes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge %Detector is used.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src.
     * @param low_threshold %Range from 0 to 100.
     *
     * <b>Note:</b>
     * The algorithm assumes that the color of the source image is close to that of the destination. This
     * assumption means that when the colors don't match, the source image color gets tinted toward the
     * color of the destination image.
     */
    public static void textureFlattening(Mat src, Mat mask, Mat dst, float low_threshold) {
        textureFlattening_2(src.nativeObj, mask.nativeObj, dst.nativeObj, low_threshold);
    }

    /**
     * By retaining only the gradients at edge locations, before integrating with the Poisson solver, one
     * washes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge %Detector is used.
     *
     * @param src Input 8-bit 3-channel image.
     * @param mask Input 8-bit 1 or 3-channel image.
     * @param dst Output image with the same size and type as src.
     *
     * <b>Note:</b>
     * The algorithm assumes that the color of the source image is close to that of the destination. This
     * assumption means that when the colors don't match, the source image color gets tinted toward the
     * color of the destination image.
     */
    public static void textureFlattening(Mat src, Mat mask, Mat dst) {
        textureFlattening_3(src.nativeObj, mask.nativeObj, dst.nativeObj);
    }




    // C++:  Ptr_AlignMTB cv::createAlignMTB(int max_bits = 6, int exclude_range = 4, bool cut = true)
    private static native long createAlignMTB_0(int max_bits, int exclude_range, boolean cut);
    private static native long createAlignMTB_1(int max_bits, int exclude_range);
    private static native long createAlignMTB_2(int max_bits);
    private static native long createAlignMTB_3();

    // C++:  Ptr_CalibrateDebevec cv::createCalibrateDebevec(int samples = 70, float lambda = 10.0f, bool random = false)
    private static native long createCalibrateDebevec_0(int samples, float lambda, boolean random);
    private static native long createCalibrateDebevec_1(int samples, float lambda);
    private static native long createCalibrateDebevec_2(int samples);
    private static native long createCalibrateDebevec_3();

    // C++:  Ptr_CalibrateRobertson cv::createCalibrateRobertson(int max_iter = 30, float threshold = 0.01f)
    private static native long createCalibrateRobertson_0(int max_iter, float threshold);
    private static native long createCalibrateRobertson_1(int max_iter);
    private static native long createCalibrateRobertson_2();

    // C++:  Ptr_MergeDebevec cv::createMergeDebevec()
    private static native long createMergeDebevec_0();

    // C++:  Ptr_MergeMertens cv::createMergeMertens(float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f)
    private static native long createMergeMertens_0(float contrast_weight, float saturation_weight, float exposure_weight);
    private static native long createMergeMertens_1(float contrast_weight, float saturation_weight);
    private static native long createMergeMertens_2(float contrast_weight);
    private static native long createMergeMertens_3();

    // C++:  Ptr_MergeRobertson cv::createMergeRobertson()
    private static native long createMergeRobertson_0();

    // C++:  Ptr_Tonemap cv::createTonemap(float gamma = 1.0f)
    private static native long createTonemap_0(float gamma);
    private static native long createTonemap_1();

    // C++:  Ptr_TonemapDrago cv::createTonemapDrago(float gamma = 1.0f, float saturation = 1.0f, float bias = 0.85f)
    private static native long createTonemapDrago_0(float gamma, float saturation, float bias);
    private static native long createTonemapDrago_1(float gamma, float saturation);
    private static native long createTonemapDrago_2(float gamma);
    private static native long createTonemapDrago_3();

    // C++:  Ptr_TonemapMantiuk cv::createTonemapMantiuk(float gamma = 1.0f, float scale = 0.7f, float saturation = 1.0f)
    private static native long createTonemapMantiuk_0(float gamma, float scale, float saturation);
    private static native long createTonemapMantiuk_1(float gamma, float scale);
    private static native long createTonemapMantiuk_2(float gamma);
    private static native long createTonemapMantiuk_3();

    // C++:  Ptr_TonemapReinhard cv::createTonemapReinhard(float gamma = 1.0f, float intensity = 0.0f, float light_adapt = 1.0f, float color_adapt = 0.0f)
    private static native long createTonemapReinhard_0(float gamma, float intensity, float light_adapt, float color_adapt);
    private static native long createTonemapReinhard_1(float gamma, float intensity, float light_adapt);
    private static native long createTonemapReinhard_2(float gamma, float intensity);
    private static native long createTonemapReinhard_3(float gamma);
    private static native long createTonemapReinhard_4();

    // C++:  void cv::colorChange(Mat src, Mat mask, Mat& dst, float red_mul = 1.0f, float green_mul = 1.0f, float blue_mul = 1.0f)
    private static native void colorChange_0(long src_nativeObj, long mask_nativeObj, long dst_nativeObj, float red_mul, float green_mul, float blue_mul);
    private static native void colorChange_1(long src_nativeObj, long mask_nativeObj, long dst_nativeObj, float red_mul, float green_mul);
    private static native void colorChange_2(long src_nativeObj, long mask_nativeObj, long dst_nativeObj, float red_mul);
    private static native void colorChange_3(long src_nativeObj, long mask_nativeObj, long dst_nativeObj);

    // C++:  void cv::decolor(Mat src, Mat& grayscale, Mat& color_boost)
    private static native void decolor_0(long src_nativeObj, long grayscale_nativeObj, long color_boost_nativeObj);

    // C++:  void cv::denoise_TVL1(vector_Mat observations, Mat result, double lambda = 1.0, int niters = 30)
    private static native void denoise_TVL1_0(long observations_mat_nativeObj, long result_nativeObj, double lambda, int niters);
    private static native void denoise_TVL1_1(long observations_mat_nativeObj, long result_nativeObj, double lambda);
    private static native void denoise_TVL1_2(long observations_mat_nativeObj, long result_nativeObj);

    // C++:  void cv::detailEnhance(Mat src, Mat& dst, float sigma_s = 10, float sigma_r = 0.15f)
    private static native void detailEnhance_0(long src_nativeObj, long dst_nativeObj, float sigma_s, float sigma_r);
    private static native void detailEnhance_1(long src_nativeObj, long dst_nativeObj, float sigma_s);
    private static native void detailEnhance_2(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::edgePreservingFilter(Mat src, Mat& dst, int flags = 1, float sigma_s = 60, float sigma_r = 0.4f)
    private static native void edgePreservingFilter_0(long src_nativeObj, long dst_nativeObj, int flags, float sigma_s, float sigma_r);
    private static native void edgePreservingFilter_1(long src_nativeObj, long dst_nativeObj, int flags, float sigma_s);
    private static native void edgePreservingFilter_2(long src_nativeObj, long dst_nativeObj, int flags);
    private static native void edgePreservingFilter_3(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::fastNlMeansDenoising(Mat src, Mat& dst, float h = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    private static native void fastNlMeansDenoising_0(long src_nativeObj, long dst_nativeObj, float h, int templateWindowSize, int searchWindowSize);
    private static native void fastNlMeansDenoising_1(long src_nativeObj, long dst_nativeObj, float h, int templateWindowSize);
    private static native void fastNlMeansDenoising_2(long src_nativeObj, long dst_nativeObj, float h);
    private static native void fastNlMeansDenoising_3(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::fastNlMeansDenoising(Mat src, Mat& dst, vector_float h, int templateWindowSize = 7, int searchWindowSize = 21, int normType = NORM_L2)
    private static native void fastNlMeansDenoising_4(long src_nativeObj, long dst_nativeObj, long h_mat_nativeObj, int templateWindowSize, int searchWindowSize, int normType);
    private static native void fastNlMeansDenoising_5(long src_nativeObj, long dst_nativeObj, long h_mat_nativeObj, int templateWindowSize, int searchWindowSize);
    private static native void fastNlMeansDenoising_6(long src_nativeObj, long dst_nativeObj, long h_mat_nativeObj, int templateWindowSize);
    private static native void fastNlMeansDenoising_7(long src_nativeObj, long dst_nativeObj, long h_mat_nativeObj);

    // C++:  void cv::fastNlMeansDenoisingColored(Mat src, Mat& dst, float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    private static native void fastNlMeansDenoisingColored_0(long src_nativeObj, long dst_nativeObj, float h, float hColor, int templateWindowSize, int searchWindowSize);
    private static native void fastNlMeansDenoisingColored_1(long src_nativeObj, long dst_nativeObj, float h, float hColor, int templateWindowSize);
    private static native void fastNlMeansDenoisingColored_2(long src_nativeObj, long dst_nativeObj, float h, float hColor);
    private static native void fastNlMeansDenoisingColored_3(long src_nativeObj, long dst_nativeObj, float h);
    private static native void fastNlMeansDenoisingColored_4(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::fastNlMeansDenoisingColoredMulti(vector_Mat srcImgs, Mat& dst, int imgToDenoiseIndex, int temporalWindowSize, float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    private static native void fastNlMeansDenoisingColoredMulti_0(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, float h, float hColor, int templateWindowSize, int searchWindowSize);
    private static native void fastNlMeansDenoisingColoredMulti_1(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, float h, float hColor, int templateWindowSize);
    private static native void fastNlMeansDenoisingColoredMulti_2(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, float h, float hColor);
    private static native void fastNlMeansDenoisingColoredMulti_3(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, float h);
    private static native void fastNlMeansDenoisingColoredMulti_4(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize);

    // C++:  void cv::fastNlMeansDenoisingMulti(vector_Mat srcImgs, Mat& dst, int imgToDenoiseIndex, int temporalWindowSize, float h = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    private static native void fastNlMeansDenoisingMulti_0(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, float h, int templateWindowSize, int searchWindowSize);
    private static native void fastNlMeansDenoisingMulti_1(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, float h, int templateWindowSize);
    private static native void fastNlMeansDenoisingMulti_2(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, float h);
    private static native void fastNlMeansDenoisingMulti_3(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize);

    // C++:  void cv::fastNlMeansDenoisingMulti(vector_Mat srcImgs, Mat& dst, int imgToDenoiseIndex, int temporalWindowSize, vector_float h, int templateWindowSize = 7, int searchWindowSize = 21, int normType = NORM_L2)
    private static native void fastNlMeansDenoisingMulti_4(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, long h_mat_nativeObj, int templateWindowSize, int searchWindowSize, int normType);
    private static native void fastNlMeansDenoisingMulti_5(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, long h_mat_nativeObj, int templateWindowSize, int searchWindowSize);
    private static native void fastNlMeansDenoisingMulti_6(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, long h_mat_nativeObj, int templateWindowSize);
    private static native void fastNlMeansDenoisingMulti_7(long srcImgs_mat_nativeObj, long dst_nativeObj, int imgToDenoiseIndex, int temporalWindowSize, long h_mat_nativeObj);

    // C++:  void cv::illuminationChange(Mat src, Mat mask, Mat& dst, float alpha = 0.2f, float beta = 0.4f)
    private static native void illuminationChange_0(long src_nativeObj, long mask_nativeObj, long dst_nativeObj, float alpha, float beta);
    private static native void illuminationChange_1(long src_nativeObj, long mask_nativeObj, long dst_nativeObj, float alpha);
    private static native void illuminationChange_2(long src_nativeObj, long mask_nativeObj, long dst_nativeObj);

    // C++:  void cv::inpaint(Mat src, Mat inpaintMask, Mat& dst, double inpaintRadius, int flags)
    private static native void inpaint_0(long src_nativeObj, long inpaintMask_nativeObj, long dst_nativeObj, double inpaintRadius, int flags);

    // C++:  void cv::pencilSketch(Mat src, Mat& dst1, Mat& dst2, float sigma_s = 60, float sigma_r = 0.07f, float shade_factor = 0.02f)
    private static native void pencilSketch_0(long src_nativeObj, long dst1_nativeObj, long dst2_nativeObj, float sigma_s, float sigma_r, float shade_factor);
    private static native void pencilSketch_1(long src_nativeObj, long dst1_nativeObj, long dst2_nativeObj, float sigma_s, float sigma_r);
    private static native void pencilSketch_2(long src_nativeObj, long dst1_nativeObj, long dst2_nativeObj, float sigma_s);
    private static native void pencilSketch_3(long src_nativeObj, long dst1_nativeObj, long dst2_nativeObj);

    // C++:  void cv::seamlessClone(Mat src, Mat dst, Mat mask, Point p, Mat& blend, int flags)
    private static native void seamlessClone_0(long src_nativeObj, long dst_nativeObj, long mask_nativeObj, double p_x, double p_y, long blend_nativeObj, int flags);

    // C++:  void cv::stylization(Mat src, Mat& dst, float sigma_s = 60, float sigma_r = 0.45f)
    private static native void stylization_0(long src_nativeObj, long dst_nativeObj, float sigma_s, float sigma_r);
    private static native void stylization_1(long src_nativeObj, long dst_nativeObj, float sigma_s);
    private static native void stylization_2(long src_nativeObj, long dst_nativeObj);

    // C++:  void cv::textureFlattening(Mat src, Mat mask, Mat& dst, float low_threshold = 30, float high_threshold = 45, int kernel_size = 3)
    private static native void textureFlattening_0(long src_nativeObj, long mask_nativeObj, long dst_nativeObj, float low_threshold, float high_threshold, int kernel_size);
    private static native void textureFlattening_1(long src_nativeObj, long mask_nativeObj, long dst_nativeObj, float low_threshold, float high_threshold);
    private static native void textureFlattening_2(long src_nativeObj, long mask_nativeObj, long dst_nativeObj, float low_threshold);
    private static native void textureFlattening_3(long src_nativeObj, long mask_nativeObj, long dst_nativeObj);

}
