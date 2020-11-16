//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.ml.ParamGrid;

// C++: class ParamGrid
/**
 * The structure represents the logarithmic grid range of statmodel parameters.
 *
 * It is used for optimizing statmodel accuracy by varying model parameters, the accuracy estimate
 * being computed by cross-validation.
 */
public class ParamGrid {

    protected final long nativeObj;
    protected ParamGrid(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static ParamGrid __fromPtr__(long addr) { return new ParamGrid(addr); }

    //
    // C++: static Ptr_ParamGrid cv::ml::ParamGrid::create(double minVal = 0., double maxVal = 0., double logstep = 1.)
    //

    /**
     * Creates a ParamGrid Ptr that can be given to the %SVM::trainAuto method
     *
     *     @param minVal minimum value of the parameter grid
     *     @param maxVal maximum value of the parameter grid
     *     @param logstep Logarithmic step for iterating the statmodel parameter
     * @return automatically generated
     */
    public static ParamGrid create(double minVal, double maxVal, double logstep) {
        return ParamGrid.__fromPtr__(create_0(minVal, maxVal, logstep));
    }

    /**
     * Creates a ParamGrid Ptr that can be given to the %SVM::trainAuto method
     *
     *     @param minVal minimum value of the parameter grid
     *     @param maxVal maximum value of the parameter grid
     * @return automatically generated
     */
    public static ParamGrid create(double minVal, double maxVal) {
        return ParamGrid.__fromPtr__(create_1(minVal, maxVal));
    }

    /**
     * Creates a ParamGrid Ptr that can be given to the %SVM::trainAuto method
     *
     *     @param minVal minimum value of the parameter grid
     * @return automatically generated
     */
    public static ParamGrid create(double minVal) {
        return ParamGrid.__fromPtr__(create_2(minVal));
    }

    /**
     * Creates a ParamGrid Ptr that can be given to the %SVM::trainAuto method
     *
     * @return automatically generated
     */
    public static ParamGrid create() {
        return ParamGrid.__fromPtr__(create_3());
    }


    //
    // C++: double ParamGrid::minVal
    //

    public double get_minVal() {
        return get_minVal_0(nativeObj);
    }


    //
    // C++: void ParamGrid::minVal
    //

    public void set_minVal(double minVal) {
        set_minVal_0(nativeObj, minVal);
    }


    //
    // C++: double ParamGrid::maxVal
    //

    public double get_maxVal() {
        return get_maxVal_0(nativeObj);
    }


    //
    // C++: void ParamGrid::maxVal
    //

    public void set_maxVal(double maxVal) {
        set_maxVal_0(nativeObj, maxVal);
    }


    //
    // C++: double ParamGrid::logStep
    //

    public double get_logStep() {
        return get_logStep_0(nativeObj);
    }


    //
    // C++: void ParamGrid::logStep
    //

    public void set_logStep(double logStep) {
        set_logStep_0(nativeObj, logStep);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_ParamGrid cv::ml::ParamGrid::create(double minVal = 0., double maxVal = 0., double logstep = 1.)
    private static native long create_0(double minVal, double maxVal, double logstep);
    private static native long create_1(double minVal, double maxVal);
    private static native long create_2(double minVal);
    private static native long create_3();

    // C++: double ParamGrid::minVal
    private static native double get_minVal_0(long nativeObj);

    // C++: void ParamGrid::minVal
    private static native void set_minVal_0(long nativeObj, double minVal);

    // C++: double ParamGrid::maxVal
    private static native double get_maxVal_0(long nativeObj);

    // C++: void ParamGrid::maxVal
    private static native void set_maxVal_0(long nativeObj, double maxVal);

    // C++: double ParamGrid::logStep
    private static native double get_logStep_0(long nativeObj);

    // C++: void ParamGrid::logStep
    private static native void set_logStep_0(long nativeObj, double logStep);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
