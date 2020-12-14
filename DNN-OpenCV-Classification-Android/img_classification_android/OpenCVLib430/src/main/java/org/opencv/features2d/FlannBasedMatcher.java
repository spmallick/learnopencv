//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FlannBasedMatcher;

// C++: class FlannBasedMatcher
/**
 * Flann-based descriptor matcher.
 *
 * This matcher trains cv::flann::Index on a train descriptor collection and calls its nearest search
 * methods to find the best matches. So, this matcher may be faster when matching a large train
 * collection than the brute force matcher. FlannBasedMatcher does not support masking permissible
 * matches of descriptor sets because flann::Index does not support this. :
 */
public class FlannBasedMatcher extends DescriptorMatcher {

    protected FlannBasedMatcher(long addr) { super(addr); }

    // internal usage only
    public static FlannBasedMatcher __fromPtr__(long addr) { return new FlannBasedMatcher(addr); }

    //
    // C++:   cv::FlannBasedMatcher::FlannBasedMatcher(Ptr_flann_IndexParams indexParams = makePtr<flann::KDTreeIndexParams>(), Ptr_flann_SearchParams searchParams = makePtr<flann::SearchParams>())
    //

    public FlannBasedMatcher() {
        super(FlannBasedMatcher_0());
    }


    //
    // C++: static Ptr_FlannBasedMatcher cv::FlannBasedMatcher::create()
    //

    public static FlannBasedMatcher create() {
        return FlannBasedMatcher.__fromPtr__(create_0());
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::FlannBasedMatcher::FlannBasedMatcher(Ptr_flann_IndexParams indexParams = makePtr<flann::KDTreeIndexParams>(), Ptr_flann_SearchParams searchParams = makePtr<flann::SearchParams>())
    private static native long FlannBasedMatcher_0();

    // C++: static Ptr_FlannBasedMatcher cv::FlannBasedMatcher::create()
    private static native long create_0();

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
