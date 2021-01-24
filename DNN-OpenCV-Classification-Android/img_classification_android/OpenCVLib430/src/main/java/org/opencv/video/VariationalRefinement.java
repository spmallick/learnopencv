//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.core.Mat;
import org.opencv.video.DenseOpticalFlow;
import org.opencv.video.VariationalRefinement;

// C++: class VariationalRefinement
/**
 * Variational optical flow refinement
 *
 * This class implements variational refinement of the input flow field, i.e.
 * it uses input flow to initialize the minimization of the following functional:
 * \(E(U) = \int_{\Omega} \delta \Psi(E_I) + \gamma \Psi(E_G) + \alpha \Psi(E_S) \),
 * where \(E_I,E_G,E_S\) are color constancy, gradient constancy and smoothness terms
 * respectively. \(\Psi(s^2)=\sqrt{s^2+\epsilon^2}\) is a robust penalizer to limit the
 * influence of outliers. A complete formulation and a description of the minimization
 * procedure can be found in CITE: Brox2004
 */
public class VariationalRefinement extends DenseOpticalFlow {

    protected VariationalRefinement(long addr) { super(addr); }

    // internal usage only
    public static VariationalRefinement __fromPtr__(long addr) { return new VariationalRefinement(addr); }

    //
    // C++: static Ptr_VariationalRefinement cv::VariationalRefinement::create()
    //

    /**
     * Creates an instance of VariationalRefinement
     * @return automatically generated
     */
    public static VariationalRefinement create() {
        return VariationalRefinement.__fromPtr__(create_0());
    }


    //
    // C++:  float cv::VariationalRefinement::getAlpha()
    //

    /**
     * Weight of the smoothness term
     * SEE: setAlpha
     * @return automatically generated
     */
    public float getAlpha() {
        return getAlpha_0(nativeObj);
    }


    //
    // C++:  float cv::VariationalRefinement::getDelta()
    //

    /**
     * Weight of the color constancy term
     * SEE: setDelta
     * @return automatically generated
     */
    public float getDelta() {
        return getDelta_0(nativeObj);
    }


    //
    // C++:  float cv::VariationalRefinement::getGamma()
    //

    /**
     * Weight of the gradient constancy term
     * SEE: setGamma
     * @return automatically generated
     */
    public float getGamma() {
        return getGamma_0(nativeObj);
    }


    //
    // C++:  float cv::VariationalRefinement::getOmega()
    //

    /**
     * Relaxation factor in SOR
     * SEE: setOmega
     * @return automatically generated
     */
    public float getOmega() {
        return getOmega_0(nativeObj);
    }


    //
    // C++:  int cv::VariationalRefinement::getFixedPointIterations()
    //

    /**
     * Number of outer (fixed-point) iterations in the minimization procedure.
     * SEE: setFixedPointIterations
     * @return automatically generated
     */
    public int getFixedPointIterations() {
        return getFixedPointIterations_0(nativeObj);
    }


    //
    // C++:  int cv::VariationalRefinement::getSorIterations()
    //

    /**
     * Number of inner successive over-relaxation (SOR) iterations
     *         in the minimization procedure to solve the respective linear system.
     * SEE: setSorIterations
     * @return automatically generated
     */
    public int getSorIterations() {
        return getSorIterations_0(nativeObj);
    }


    //
    // C++:  void cv::VariationalRefinement::calcUV(Mat I0, Mat I1, Mat& flow_u, Mat& flow_v)
    //

    /**
     * REF: calc function overload to handle separate horizontal (u) and vertical (v) flow components
     * (to avoid extra splits/merges)
     * @param I0 automatically generated
     * @param I1 automatically generated
     * @param flow_u automatically generated
     * @param flow_v automatically generated
     */
    public void calcUV(Mat I0, Mat I1, Mat flow_u, Mat flow_v) {
        calcUV_0(nativeObj, I0.nativeObj, I1.nativeObj, flow_u.nativeObj, flow_v.nativeObj);
    }


    //
    // C++:  void cv::VariationalRefinement::setAlpha(float val)
    //

    /**
     *  getAlpha SEE: getAlpha
     * @param val automatically generated
     */
    public void setAlpha(float val) {
        setAlpha_0(nativeObj, val);
    }


    //
    // C++:  void cv::VariationalRefinement::setDelta(float val)
    //

    /**
     *  getDelta SEE: getDelta
     * @param val automatically generated
     */
    public void setDelta(float val) {
        setDelta_0(nativeObj, val);
    }


    //
    // C++:  void cv::VariationalRefinement::setFixedPointIterations(int val)
    //

    /**
     *  getFixedPointIterations SEE: getFixedPointIterations
     * @param val automatically generated
     */
    public void setFixedPointIterations(int val) {
        setFixedPointIterations_0(nativeObj, val);
    }


    //
    // C++:  void cv::VariationalRefinement::setGamma(float val)
    //

    /**
     *  getGamma SEE: getGamma
     * @param val automatically generated
     */
    public void setGamma(float val) {
        setGamma_0(nativeObj, val);
    }


    //
    // C++:  void cv::VariationalRefinement::setOmega(float val)
    //

    /**
     *  getOmega SEE: getOmega
     * @param val automatically generated
     */
    public void setOmega(float val) {
        setOmega_0(nativeObj, val);
    }


    //
    // C++:  void cv::VariationalRefinement::setSorIterations(int val)
    //

    /**
     *  getSorIterations SEE: getSorIterations
     * @param val automatically generated
     */
    public void setSorIterations(int val) {
        setSorIterations_0(nativeObj, val);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_VariationalRefinement cv::VariationalRefinement::create()
    private static native long create_0();

    // C++:  float cv::VariationalRefinement::getAlpha()
    private static native float getAlpha_0(long nativeObj);

    // C++:  float cv::VariationalRefinement::getDelta()
    private static native float getDelta_0(long nativeObj);

    // C++:  float cv::VariationalRefinement::getGamma()
    private static native float getGamma_0(long nativeObj);

    // C++:  float cv::VariationalRefinement::getOmega()
    private static native float getOmega_0(long nativeObj);

    // C++:  int cv::VariationalRefinement::getFixedPointIterations()
    private static native int getFixedPointIterations_0(long nativeObj);

    // C++:  int cv::VariationalRefinement::getSorIterations()
    private static native int getSorIterations_0(long nativeObj);

    // C++:  void cv::VariationalRefinement::calcUV(Mat I0, Mat I1, Mat& flow_u, Mat& flow_v)
    private static native void calcUV_0(long nativeObj, long I0_nativeObj, long I1_nativeObj, long flow_u_nativeObj, long flow_v_nativeObj);

    // C++:  void cv::VariationalRefinement::setAlpha(float val)
    private static native void setAlpha_0(long nativeObj, float val);

    // C++:  void cv::VariationalRefinement::setDelta(float val)
    private static native void setDelta_0(long nativeObj, float val);

    // C++:  void cv::VariationalRefinement::setFixedPointIterations(int val)
    private static native void setFixedPointIterations_0(long nativeObj, int val);

    // C++:  void cv::VariationalRefinement::setGamma(float val)
    private static native void setGamma_0(long nativeObj, float val);

    // C++:  void cv::VariationalRefinement::setOmega(float val)
    private static native void setOmega_0(long nativeObj, float val);

    // C++:  void cv::VariationalRefinement::setSorIterations(int val)
    private static native void setSorIterations_0(long nativeObj, int val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
