//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;

// C++: class BFMatcher
/**
 * Brute-force descriptor matcher.
 *
 * For each descriptor in the first set, this matcher finds the closest descriptor in the second set
 * by trying each one. This descriptor matcher supports masking permissible matches of descriptor
 * sets.
 */
public class BFMatcher extends DescriptorMatcher {

    protected BFMatcher(long addr) { super(addr); }

    // internal usage only
    public static BFMatcher __fromPtr__(long addr) { return new BFMatcher(addr); }

    //
    // C++:   cv::BFMatcher::BFMatcher(int normType = NORM_L2, bool crossCheck = false)
    //

    /**
     * Brute-force matcher constructor (obsolete). Please use BFMatcher.create()
     *
     *
     * @param normType automatically generated
     * @param crossCheck automatically generated
     */
    public BFMatcher(int normType, boolean crossCheck) {
        super(BFMatcher_0(normType, crossCheck));
    }

    /**
     * Brute-force matcher constructor (obsolete). Please use BFMatcher.create()
     *
     *
     * @param normType automatically generated
     */
    public BFMatcher(int normType) {
        super(BFMatcher_1(normType));
    }

    /**
     * Brute-force matcher constructor (obsolete). Please use BFMatcher.create()
     *
     *
     */
    public BFMatcher() {
        super(BFMatcher_2());
    }


    //
    // C++: static Ptr_BFMatcher cv::BFMatcher::create(int normType = NORM_L2, bool crossCheck = false)
    //

    /**
     * Brute-force matcher create method.
     *     @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
     *     preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
     *     BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
     *     description).
     *     @param crossCheck If it is false, this is will be default BFMatcher behaviour when it finds the k
     *     nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with
     *     k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
     *     matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent
     *     pairs. Such technique usually produces best results with minimal number of outliers when there are
     *     enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
     * @return automatically generated
     */
    public static BFMatcher create(int normType, boolean crossCheck) {
        return BFMatcher.__fromPtr__(create_0(normType, crossCheck));
    }

    /**
     * Brute-force matcher create method.
     *     @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
     *     preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
     *     BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
     *     description).
     *     nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with
     *     k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
     *     matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent
     *     pairs. Such technique usually produces best results with minimal number of outliers when there are
     *     enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
     * @return automatically generated
     */
    public static BFMatcher create(int normType) {
        return BFMatcher.__fromPtr__(create_1(normType));
    }

    /**
     * Brute-force matcher create method.
     *     preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
     *     BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
     *     description).
     *     nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with
     *     k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
     *     matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent
     *     pairs. Such technique usually produces best results with minimal number of outliers when there are
     *     enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
     * @return automatically generated
     */
    public static BFMatcher create() {
        return BFMatcher.__fromPtr__(create_2());
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::BFMatcher::BFMatcher(int normType = NORM_L2, bool crossCheck = false)
    private static native long BFMatcher_0(int normType, boolean crossCheck);
    private static native long BFMatcher_1(int normType);
    private static native long BFMatcher_2();

    // C++: static Ptr_BFMatcher cv::BFMatcher::create(int normType = NORM_L2, bool crossCheck = false)
    private static native long create_0(int normType, boolean crossCheck);
    private static native long create_1(int normType);
    private static native long create_2();

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
