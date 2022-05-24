//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.core;



// C++: class TickMeter
/**
 * a Class to measure passing time.
 *
 * The class computes passing time by counting the number of ticks per second. That is, the following code computes the
 * execution time in seconds:
 * <code>
 * TickMeter tm;
 * tm.start();
 * // do something ...
 * tm.stop();
 * std::cout &lt;&lt; tm.getTimeSec();
 * </code>
 *
 * It is also possible to compute the average time over multiple runs:
 * <code>
 * TickMeter tm;
 * for (int i = 0; i &lt; 100; i++)
 * {
 *     tm.start();
 *     // do something ...
 *     tm.stop();
 * }
 * double average_time = tm.getTimeSec() / tm.getCounter();
 * std::cout &lt;&lt; "Average time in second per iteration is: " &lt;&lt; average_time &lt;&lt; std::endl;
 * </code>
 * SEE: getTickCount, getTickFrequency
 */
public class TickMeter {

    protected final long nativeObj;
    protected TickMeter(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static TickMeter __fromPtr__(long addr) { return new TickMeter(addr); }

    //
    // C++:   cv::TickMeter::TickMeter()
    //

    public TickMeter() {
        nativeObj = TickMeter_0();
    }


    //
    // C++:  double cv::TickMeter::getTimeMicro()
    //

    /**
     * returns passed time in microseconds.
     * @return automatically generated
     */
    public double getTimeMicro() {
        return getTimeMicro_0(nativeObj);
    }


    //
    // C++:  double cv::TickMeter::getTimeMilli()
    //

    /**
     * returns passed time in milliseconds.
     * @return automatically generated
     */
    public double getTimeMilli() {
        return getTimeMilli_0(nativeObj);
    }


    //
    // C++:  double cv::TickMeter::getTimeSec()
    //

    /**
     * returns passed time in seconds.
     * @return automatically generated
     */
    public double getTimeSec() {
        return getTimeSec_0(nativeObj);
    }


    //
    // C++:  int64 cv::TickMeter::getCounter()
    //

    /**
     * returns internal counter value.
     * @return automatically generated
     */
    public long getCounter() {
        return getCounter_0(nativeObj);
    }


    //
    // C++:  int64 cv::TickMeter::getTimeTicks()
    //

    /**
     * returns counted ticks.
     * @return automatically generated
     */
    public long getTimeTicks() {
        return getTimeTicks_0(nativeObj);
    }


    //
    // C++:  void cv::TickMeter::reset()
    //

    /**
     * resets internal values.
     */
    public void reset() {
        reset_0(nativeObj);
    }


    //
    // C++:  void cv::TickMeter::start()
    //

    /**
     * starts counting ticks.
     */
    public void start() {
        start_0(nativeObj);
    }


    //
    // C++:  void cv::TickMeter::stop()
    //

    /**
     * stops counting ticks.
     */
    public void stop() {
        stop_0(nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::TickMeter::TickMeter()
    private static native long TickMeter_0();

    // C++:  double cv::TickMeter::getTimeMicro()
    private static native double getTimeMicro_0(long nativeObj);

    // C++:  double cv::TickMeter::getTimeMilli()
    private static native double getTimeMilli_0(long nativeObj);

    // C++:  double cv::TickMeter::getTimeSec()
    private static native double getTimeSec_0(long nativeObj);

    // C++:  int64 cv::TickMeter::getCounter()
    private static native long getCounter_0(long nativeObj);

    // C++:  int64 cv::TickMeter::getTimeTicks()
    private static native long getTimeTicks_0(long nativeObj);

    // C++:  void cv::TickMeter::reset()
    private static native void reset_0(long nativeObj);

    // C++:  void cv::TickMeter::start()
    private static native void start_0(long nativeObj);

    // C++:  void cv::TickMeter::stop()
    private static native void stop_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
