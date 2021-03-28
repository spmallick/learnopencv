//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.objdetect;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.utils.Converters;

// C++: class Objdetect

public class Objdetect {

    // C++: enum <unnamed>
    public static final int
            CASCADE_DO_CANNY_PRUNING = 1,
            CASCADE_SCALE_IMAGE = 2,
            CASCADE_FIND_BIGGEST_OBJECT = 4,
            CASCADE_DO_ROUGH_SEARCH = 8;


    // C++: enum ObjectStatus
    public static final int
            DetectionBasedTracker_DETECTED_NOT_SHOWN_YET = 0,
            DetectionBasedTracker_DETECTED = 1,
            DetectionBasedTracker_DETECTED_TEMPORARY_LOST = 2,
            DetectionBasedTracker_WRONG_OBJECT = 3;


    //
    // C++:  void cv::groupRectangles(vector_Rect& rectList, vector_int& weights, int groupThreshold, double eps = 0.2)
    //

    public static void groupRectangles(MatOfRect rectList, MatOfInt weights, int groupThreshold, double eps) {
        Mat rectList_mat = rectList;
        Mat weights_mat = weights;
        groupRectangles_0(rectList_mat.nativeObj, weights_mat.nativeObj, groupThreshold, eps);
    }

    public static void groupRectangles(MatOfRect rectList, MatOfInt weights, int groupThreshold) {
        Mat rectList_mat = rectList;
        Mat weights_mat = weights;
        groupRectangles_1(rectList_mat.nativeObj, weights_mat.nativeObj, groupThreshold);
    }




    // C++:  void cv::groupRectangles(vector_Rect& rectList, vector_int& weights, int groupThreshold, double eps = 0.2)
    private static native void groupRectangles_0(long rectList_mat_nativeObj, long weights_mat_nativeObj, int groupThreshold, double eps);
    private static native void groupRectangles_1(long rectList_mat_nativeObj, long weights_mat_nativeObj, int groupThreshold);

}
