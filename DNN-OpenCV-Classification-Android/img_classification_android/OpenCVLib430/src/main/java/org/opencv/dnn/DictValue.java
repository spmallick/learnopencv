//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.dnn;



// C++: class DictValue
/**
 * This struct stores the scalar value (or array) of one of the following type: double, cv::String or int64.
 * TODO: Maybe int64 is useless because double type exactly stores at least 2^52 integers.
 */
public class DictValue {

    protected final long nativeObj;
    protected DictValue(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static DictValue __fromPtr__(long addr) { return new DictValue(addr); }

    //
    // C++:   cv::dnn::DictValue::DictValue(String s)
    //

    public DictValue(String s) {
        nativeObj = DictValue_0(s);
    }


    //
    // C++:   cv::dnn::DictValue::DictValue(double p)
    //

    public DictValue(double p) {
        nativeObj = DictValue_1(p);
    }


    //
    // C++:   cv::dnn::DictValue::DictValue(int i)
    //

    public DictValue(int i) {
        nativeObj = DictValue_2(i);
    }


    //
    // C++:  String cv::dnn::DictValue::getStringValue(int idx = -1)
    //

    public String getStringValue(int idx) {
        return getStringValue_0(nativeObj, idx);
    }

    public String getStringValue() {
        return getStringValue_1(nativeObj);
    }


    //
    // C++:  bool cv::dnn::DictValue::isInt()
    //

    public boolean isInt() {
        return isInt_0(nativeObj);
    }


    //
    // C++:  bool cv::dnn::DictValue::isReal()
    //

    public boolean isReal() {
        return isReal_0(nativeObj);
    }


    //
    // C++:  bool cv::dnn::DictValue::isString()
    //

    public boolean isString() {
        return isString_0(nativeObj);
    }


    //
    // C++:  double cv::dnn::DictValue::getRealValue(int idx = -1)
    //

    public double getRealValue(int idx) {
        return getRealValue_0(nativeObj, idx);
    }

    public double getRealValue() {
        return getRealValue_1(nativeObj);
    }


    //
    // C++:  int cv::dnn::DictValue::getIntValue(int idx = -1)
    //

    public int getIntValue(int idx) {
        return getIntValue_0(nativeObj, idx);
    }

    public int getIntValue() {
        return getIntValue_1(nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::dnn::DictValue::DictValue(String s)
    private static native long DictValue_0(String s);

    // C++:   cv::dnn::DictValue::DictValue(double p)
    private static native long DictValue_1(double p);

    // C++:   cv::dnn::DictValue::DictValue(int i)
    private static native long DictValue_2(int i);

    // C++:  String cv::dnn::DictValue::getStringValue(int idx = -1)
    private static native String getStringValue_0(long nativeObj, int idx);
    private static native String getStringValue_1(long nativeObj);

    // C++:  bool cv::dnn::DictValue::isInt()
    private static native boolean isInt_0(long nativeObj);

    // C++:  bool cv::dnn::DictValue::isReal()
    private static native boolean isReal_0(long nativeObj);

    // C++:  bool cv::dnn::DictValue::isString()
    private static native boolean isString_0(long nativeObj);

    // C++:  double cv::dnn::DictValue::getRealValue(int idx = -1)
    private static native double getRealValue_0(long nativeObj, int idx);
    private static native double getRealValue_1(long nativeObj);

    // C++:  int cv::dnn::DictValue::getIntValue(int idx = -1)
    private static native int getIntValue_0(long nativeObj, int idx);
    private static native int getIntValue_1(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
