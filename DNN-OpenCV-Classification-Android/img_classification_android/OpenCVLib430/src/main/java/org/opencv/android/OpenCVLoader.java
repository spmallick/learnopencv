package org.opencv.android;

import android.content.Context;

/**
 * Helper class provides common initialization methods for OpenCV library.
 */
public class OpenCVLoader
{
    /**
     * OpenCV Library version 2.4.2.
     */
    public static final String OPENCV_VERSION_2_4_2 = "2.4.2";

    /**
     * OpenCV Library version 2.4.3.
     */
    public static final String OPENCV_VERSION_2_4_3 = "2.4.3";

    /**
     * OpenCV Library version 2.4.4.
     */
    public static final String OPENCV_VERSION_2_4_4 = "2.4.4";

    /**
     * OpenCV Library version 2.4.5.
     */
    public static final String OPENCV_VERSION_2_4_5 = "2.4.5";

    /**
     * OpenCV Library version 2.4.6.
     */
    public static final String OPENCV_VERSION_2_4_6 = "2.4.6";

    /**
     * OpenCV Library version 2.4.7.
     */
    public static final String OPENCV_VERSION_2_4_7 = "2.4.7";

    /**
     * OpenCV Library version 2.4.8.
     */
    public static final String OPENCV_VERSION_2_4_8 = "2.4.8";

    /**
     * OpenCV Library version 2.4.9.
     */
    public static final String OPENCV_VERSION_2_4_9 = "2.4.9";

    /**
     * OpenCV Library version 2.4.10.
     */
    public static final String OPENCV_VERSION_2_4_10 = "2.4.10";

    /**
     * OpenCV Library version 2.4.11.
     */
    public static final String OPENCV_VERSION_2_4_11 = "2.4.11";

    /**
     * OpenCV Library version 2.4.12.
     */
    public static final String OPENCV_VERSION_2_4_12 = "2.4.12";

    /**
     * OpenCV Library version 2.4.13.
     */
    public static final String OPENCV_VERSION_2_4_13 = "2.4.13";

    /**
     * OpenCV Library version 3.0.0.
     */
    public static final String OPENCV_VERSION_3_0_0 = "3.0.0";

    /**
     * OpenCV Library version 3.1.0.
     */
    public static final String OPENCV_VERSION_3_1_0 = "3.1.0";

    /**
     * OpenCV Library version 3.2.0.
     */
    public static final String OPENCV_VERSION_3_2_0 = "3.2.0";

    /**
     * OpenCV Library version 3.3.0.
     */
    public static final String OPENCV_VERSION_3_3_0 = "3.3.0";

    /**
     * OpenCV Library version 3.4.0.
     */
    public static final String OPENCV_VERSION_3_4_0 = "3.4.0";

    /**
     * Current OpenCV Library version
     */
    public static final String OPENCV_VERSION = "4.3.0";


    /**
     * Loads and initializes OpenCV library from current application package. Roughly, it's an analog of system.loadLibrary("opencv_java").
     * @return Returns true is initialization of OpenCV was successful.
     */
    public static boolean initDebug()
    {
        return StaticHelper.initOpenCV(false);
    }

    /**
     * Loads and initializes OpenCV library from current application package. Roughly, it's an analog of system.loadLibrary("opencv_java").
     * @param InitCuda load and initialize CUDA runtime libraries.
     * @return Returns true is initialization of OpenCV was successful.
     */
    public static boolean initDebug(boolean InitCuda)
    {
        return StaticHelper.initOpenCV(InitCuda);
    }

    /**
     * Loads and initializes OpenCV library using OpenCV Engine service.
     * @param Version OpenCV library version.
     * @param AppContext application context for connecting to the service.
     * @param Callback object, that implements LoaderCallbackInterface for handling the connection status.
     * @return Returns true if initialization of OpenCV is successful.
     */
    public static boolean initAsync(String Version, Context AppContext,
            LoaderCallbackInterface Callback)
    {
        return AsyncServiceHelper.initOpenCV(Version, AppContext, Callback);
    }
}
