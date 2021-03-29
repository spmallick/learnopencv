//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.utils.Converters;

// C++: class DescriptorMatcher
/**
 * Abstract base class for matching keypoint descriptors.
 *
 * It has two groups of match methods: for matching descriptors of an image with another image or with
 * an image set.
 */
public class DescriptorMatcher extends Algorithm {

    protected DescriptorMatcher(long addr) { super(addr); }

    // internal usage only
    public static DescriptorMatcher __fromPtr__(long addr) { return new DescriptorMatcher(addr); }

    // C++: enum MatcherType
    public static final int
            FLANNBASED = 1,
            BRUTEFORCE = 2,
            BRUTEFORCE_L1 = 3,
            BRUTEFORCE_HAMMING = 4,
            BRUTEFORCE_HAMMINGLUT = 5,
            BRUTEFORCE_SL2 = 6;


    //
    // C++:  Ptr_DescriptorMatcher cv::DescriptorMatcher::clone(bool emptyTrainData = false)
    //

    /**
     * Clones the matcher.
     *
     *     @param emptyTrainData If emptyTrainData is false, the method creates a deep copy of the object,
     *     that is, copies both parameters and train data. If emptyTrainData is true, the method creates an
     *     object copy with the current parameters but with empty train data.
     * @return automatically generated
     */
    public DescriptorMatcher clone(boolean emptyTrainData) {
        return DescriptorMatcher.__fromPtr__(clone_0(nativeObj, emptyTrainData));
    }

    /**
     * Clones the matcher.
     *
     *     that is, copies both parameters and train data. If emptyTrainData is true, the method creates an
     *     object copy with the current parameters but with empty train data.
     * @return automatically generated
     */
    public DescriptorMatcher clone() {
        return DescriptorMatcher.__fromPtr__(clone_1(nativeObj));
    }


    //
    // C++: static Ptr_DescriptorMatcher cv::DescriptorMatcher::create(DescriptorMatcher_MatcherType matcherType)
    //

    public static DescriptorMatcher create(int matcherType) {
        return DescriptorMatcher.__fromPtr__(create_0(matcherType));
    }


    //
    // C++: static Ptr_DescriptorMatcher cv::DescriptorMatcher::create(String descriptorMatcherType)
    //

    /**
     * Creates a descriptor matcher of a given type with the default parameters (using default
     *     constructor).
     *
     *     @param descriptorMatcherType Descriptor matcher type. Now the following matcher types are
     *     supported:
     * <ul>
     *   <li>
     *        {@code BruteForce} (it uses L2 )
     *   </li>
     *   <li>
     *        {@code BruteForce-L1}
     *   </li>
     *   <li>
     *        {@code BruteForce-Hamming}
     *   </li>
     *   <li>
     *        {@code BruteForce-Hamming(2)}
     *   </li>
     *   <li>
     *        {@code FlannBased}
     *   </li>
     * </ul>
     * @return automatically generated
     */
    public static DescriptorMatcher create(String descriptorMatcherType) {
        return DescriptorMatcher.__fromPtr__(create_1(descriptorMatcherType));
    }


    //
    // C++:  bool cv::DescriptorMatcher::empty()
    //

    /**
     * Returns true if there are no train descriptors in the both collections.
     * @return automatically generated
     */
    public boolean empty() {
        return empty_0(nativeObj);
    }


    //
    // C++:  bool cv::DescriptorMatcher::isMaskSupported()
    //

    /**
     * Returns true if the descriptor matcher supports masking permissible matches.
     * @return automatically generated
     */
    public boolean isMaskSupported() {
        return isMaskSupported_0(nativeObj);
    }


    //
    // C++:  vector_Mat cv::DescriptorMatcher::getTrainDescriptors()
    //

    /**
     * Returns a constant link to the train descriptor collection trainDescCollection .
     * @return automatically generated
     */
    public List<Mat> getTrainDescriptors() {
        List<Mat> retVal = new ArrayList<Mat>();
        Mat retValMat = new Mat(getTrainDescriptors_0(nativeObj));
        Converters.Mat_to_vector_Mat(retValMat, retVal);
        return retVal;
    }


    //
    // C++:  void cv::DescriptorMatcher::add(vector_Mat descriptors)
    //

    /**
     * Adds descriptors to train a CPU(trainDescCollectionis) or GPU(utrainDescCollectionis) descriptor
     *     collection.
     *
     *     If the collection is not empty, the new descriptors are added to existing train descriptors.
     *
     *     @param descriptors Descriptors to add. Each descriptors[i] is a set of descriptors from the same
     *     train image.
     */
    public void add(List<Mat> descriptors) {
        Mat descriptors_mat = Converters.vector_Mat_to_Mat(descriptors);
        add_0(nativeObj, descriptors_mat.nativeObj);
    }


    //
    // C++:  void cv::DescriptorMatcher::clear()
    //

    /**
     * Clears the train descriptor collections.
     */
    public void clear() {
        clear_0(nativeObj);
    }


    //
    // C++:  void cv::DescriptorMatcher::knnMatch(Mat queryDescriptors, Mat trainDescriptors, vector_vector_DMatch& matches, int k, Mat mask = Mat(), bool compactResult = false)
    //

    /**
     * Finds the k best matches for each descriptor from a query set.
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
     *     collection stored in the class object.
     *     @param mask Mask specifying permissible matches between an input query and train matrices of
     *     descriptors.
     *     @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
     *     @param k Count of best matches found per each query descriptor or less if a query descriptor has
     *     less than k possible matches in total.
     *     @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     *
     *     These extended variants of DescriptorMatcher::match methods find several best matches for each query
     *     descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match
     *     for the details about query and train descriptors.
     */
    public void knnMatch(Mat queryDescriptors, Mat trainDescriptors, List<MatOfDMatch> matches, int k, Mat mask, boolean compactResult) {
        Mat matches_mat = new Mat();
        knnMatch_0(nativeObj, queryDescriptors.nativeObj, trainDescriptors.nativeObj, matches_mat.nativeObj, k, mask.nativeObj, compactResult);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }

    /**
     * Finds the k best matches for each descriptor from a query set.
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
     *     collection stored in the class object.
     *     @param mask Mask specifying permissible matches between an input query and train matrices of
     *     descriptors.
     *     @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
     *     @param k Count of best matches found per each query descriptor or less if a query descriptor has
     *     less than k possible matches in total.
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     *
     *     These extended variants of DescriptorMatcher::match methods find several best matches for each query
     *     descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match
     *     for the details about query and train descriptors.
     */
    public void knnMatch(Mat queryDescriptors, Mat trainDescriptors, List<MatOfDMatch> matches, int k, Mat mask) {
        Mat matches_mat = new Mat();
        knnMatch_1(nativeObj, queryDescriptors.nativeObj, trainDescriptors.nativeObj, matches_mat.nativeObj, k, mask.nativeObj);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }

    /**
     * Finds the k best matches for each descriptor from a query set.
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
     *     collection stored in the class object.
     *     descriptors.
     *     @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
     *     @param k Count of best matches found per each query descriptor or less if a query descriptor has
     *     less than k possible matches in total.
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     *
     *     These extended variants of DescriptorMatcher::match methods find several best matches for each query
     *     descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match
     *     for the details about query and train descriptors.
     */
    public void knnMatch(Mat queryDescriptors, Mat trainDescriptors, List<MatOfDMatch> matches, int k) {
        Mat matches_mat = new Mat();
        knnMatch_2(nativeObj, queryDescriptors.nativeObj, trainDescriptors.nativeObj, matches_mat.nativeObj, k);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }


    //
    // C++:  void cv::DescriptorMatcher::knnMatch(Mat queryDescriptors, vector_vector_DMatch& matches, int k, vector_Mat masks = vector_Mat(), bool compactResult = false)
    //

    /**
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
     *     @param k Count of best matches found per each query descriptor or less if a query descriptor has
     *     less than k possible matches in total.
     *     @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
     *     descriptors and stored train descriptors from the i-th image trainDescCollection[i].
     *     @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     */
    public void knnMatch(Mat queryDescriptors, List<MatOfDMatch> matches, int k, List<Mat> masks, boolean compactResult) {
        Mat matches_mat = new Mat();
        Mat masks_mat = Converters.vector_Mat_to_Mat(masks);
        knnMatch_3(nativeObj, queryDescriptors.nativeObj, matches_mat.nativeObj, k, masks_mat.nativeObj, compactResult);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }

    /**
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
     *     @param k Count of best matches found per each query descriptor or less if a query descriptor has
     *     less than k possible matches in total.
     *     @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
     *     descriptors and stored train descriptors from the i-th image trainDescCollection[i].
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     */
    public void knnMatch(Mat queryDescriptors, List<MatOfDMatch> matches, int k, List<Mat> masks) {
        Mat matches_mat = new Mat();
        Mat masks_mat = Converters.vector_Mat_to_Mat(masks);
        knnMatch_4(nativeObj, queryDescriptors.nativeObj, matches_mat.nativeObj, k, masks_mat.nativeObj);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }

    /**
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
     *     @param k Count of best matches found per each query descriptor or less if a query descriptor has
     *     less than k possible matches in total.
     *     descriptors and stored train descriptors from the i-th image trainDescCollection[i].
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     */
    public void knnMatch(Mat queryDescriptors, List<MatOfDMatch> matches, int k) {
        Mat matches_mat = new Mat();
        knnMatch_5(nativeObj, queryDescriptors.nativeObj, matches_mat.nativeObj, k);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }


    //
    // C++:  void cv::DescriptorMatcher::match(Mat queryDescriptors, Mat trainDescriptors, vector_DMatch& matches, Mat mask = Mat())
    //

    /**
     * Finds the best match for each descriptor from a query set.
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
     *     collection stored in the class object.
     *     @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
     *     descriptor. So, matches size may be smaller than the query descriptors count.
     *     @param mask Mask specifying permissible matches between an input query and train matrices of
     *     descriptors.
     *
     *     In the first variant of this method, the train descriptors are passed as an input argument. In the
     *     second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is
     *     used. Optional mask (or masks) can be passed to specify which query and training descriptors can be
     *     matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if
     *     mask.at&lt;uchar&gt;(i,j) is non-zero.
     */
    public void match(Mat queryDescriptors, Mat trainDescriptors, MatOfDMatch matches, Mat mask) {
        Mat matches_mat = matches;
        match_0(nativeObj, queryDescriptors.nativeObj, trainDescriptors.nativeObj, matches_mat.nativeObj, mask.nativeObj);
    }

    /**
     * Finds the best match for each descriptor from a query set.
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
     *     collection stored in the class object.
     *     @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
     *     descriptor. So, matches size may be smaller than the query descriptors count.
     *     descriptors.
     *
     *     In the first variant of this method, the train descriptors are passed as an input argument. In the
     *     second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is
     *     used. Optional mask (or masks) can be passed to specify which query and training descriptors can be
     *     matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if
     *     mask.at&lt;uchar&gt;(i,j) is non-zero.
     */
    public void match(Mat queryDescriptors, Mat trainDescriptors, MatOfDMatch matches) {
        Mat matches_mat = matches;
        match_1(nativeObj, queryDescriptors.nativeObj, trainDescriptors.nativeObj, matches_mat.nativeObj);
    }


    //
    // C++:  void cv::DescriptorMatcher::match(Mat queryDescriptors, vector_DMatch& matches, vector_Mat masks = vector_Mat())
    //

    /**
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
     *     descriptor. So, matches size may be smaller than the query descriptors count.
     *     @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
     *     descriptors and stored train descriptors from the i-th image trainDescCollection[i].
     */
    public void match(Mat queryDescriptors, MatOfDMatch matches, List<Mat> masks) {
        Mat matches_mat = matches;
        Mat masks_mat = Converters.vector_Mat_to_Mat(masks);
        match_2(nativeObj, queryDescriptors.nativeObj, matches_mat.nativeObj, masks_mat.nativeObj);
    }

    /**
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
     *     descriptor. So, matches size may be smaller than the query descriptors count.
     *     descriptors and stored train descriptors from the i-th image trainDescCollection[i].
     */
    public void match(Mat queryDescriptors, MatOfDMatch matches) {
        Mat matches_mat = matches;
        match_3(nativeObj, queryDescriptors.nativeObj, matches_mat.nativeObj);
    }


    //
    // C++:  void cv::DescriptorMatcher::radiusMatch(Mat queryDescriptors, Mat trainDescriptors, vector_vector_DMatch& matches, float maxDistance, Mat mask = Mat(), bool compactResult = false)
    //

    /**
     * For each query descriptor, finds the training descriptors not farther than the specified distance.
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
     *     collection stored in the class object.
     *     @param matches Found matches.
     *     @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     *     @param maxDistance Threshold for the distance between matched descriptors. Distance means here
     *     metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
     *     in Pixels)!
     *     @param mask Mask specifying permissible matches between an input query and train matrices of
     *     descriptors.
     *
     *     For each query descriptor, the methods find such training descriptors that the distance between the
     *     query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are
     *     returned in the distance increasing order.
     */
    public void radiusMatch(Mat queryDescriptors, Mat trainDescriptors, List<MatOfDMatch> matches, float maxDistance, Mat mask, boolean compactResult) {
        Mat matches_mat = new Mat();
        radiusMatch_0(nativeObj, queryDescriptors.nativeObj, trainDescriptors.nativeObj, matches_mat.nativeObj, maxDistance, mask.nativeObj, compactResult);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }

    /**
     * For each query descriptor, finds the training descriptors not farther than the specified distance.
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
     *     collection stored in the class object.
     *     @param matches Found matches.
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     *     @param maxDistance Threshold for the distance between matched descriptors. Distance means here
     *     metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
     *     in Pixels)!
     *     @param mask Mask specifying permissible matches between an input query and train matrices of
     *     descriptors.
     *
     *     For each query descriptor, the methods find such training descriptors that the distance between the
     *     query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are
     *     returned in the distance increasing order.
     */
    public void radiusMatch(Mat queryDescriptors, Mat trainDescriptors, List<MatOfDMatch> matches, float maxDistance, Mat mask) {
        Mat matches_mat = new Mat();
        radiusMatch_1(nativeObj, queryDescriptors.nativeObj, trainDescriptors.nativeObj, matches_mat.nativeObj, maxDistance, mask.nativeObj);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }

    /**
     * For each query descriptor, finds the training descriptors not farther than the specified distance.
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
     *     collection stored in the class object.
     *     @param matches Found matches.
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     *     @param maxDistance Threshold for the distance between matched descriptors. Distance means here
     *     metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
     *     in Pixels)!
     *     descriptors.
     *
     *     For each query descriptor, the methods find such training descriptors that the distance between the
     *     query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are
     *     returned in the distance increasing order.
     */
    public void radiusMatch(Mat queryDescriptors, Mat trainDescriptors, List<MatOfDMatch> matches, float maxDistance) {
        Mat matches_mat = new Mat();
        radiusMatch_2(nativeObj, queryDescriptors.nativeObj, trainDescriptors.nativeObj, matches_mat.nativeObj, maxDistance);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }


    //
    // C++:  void cv::DescriptorMatcher::radiusMatch(Mat queryDescriptors, vector_vector_DMatch& matches, float maxDistance, vector_Mat masks = vector_Mat(), bool compactResult = false)
    //

    /**
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param matches Found matches.
     *     @param maxDistance Threshold for the distance between matched descriptors. Distance means here
     *     metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
     *     in Pixels)!
     *     @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
     *     descriptors and stored train descriptors from the i-th image trainDescCollection[i].
     *     @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     */
    public void radiusMatch(Mat queryDescriptors, List<MatOfDMatch> matches, float maxDistance, List<Mat> masks, boolean compactResult) {
        Mat matches_mat = new Mat();
        Mat masks_mat = Converters.vector_Mat_to_Mat(masks);
        radiusMatch_3(nativeObj, queryDescriptors.nativeObj, matches_mat.nativeObj, maxDistance, masks_mat.nativeObj, compactResult);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }

    /**
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param matches Found matches.
     *     @param maxDistance Threshold for the distance between matched descriptors. Distance means here
     *     metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
     *     in Pixels)!
     *     @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
     *     descriptors and stored train descriptors from the i-th image trainDescCollection[i].
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     */
    public void radiusMatch(Mat queryDescriptors, List<MatOfDMatch> matches, float maxDistance, List<Mat> masks) {
        Mat matches_mat = new Mat();
        Mat masks_mat = Converters.vector_Mat_to_Mat(masks);
        radiusMatch_4(nativeObj, queryDescriptors.nativeObj, matches_mat.nativeObj, maxDistance, masks_mat.nativeObj);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }

    /**
     *
     *     @param queryDescriptors Query set of descriptors.
     *     @param matches Found matches.
     *     @param maxDistance Threshold for the distance between matched descriptors. Distance means here
     *     metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
     *     in Pixels)!
     *     descriptors and stored train descriptors from the i-th image trainDescCollection[i].
     *     false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
     *     the matches vector does not contain matches for fully masked-out query descriptors.
     */
    public void radiusMatch(Mat queryDescriptors, List<MatOfDMatch> matches, float maxDistance) {
        Mat matches_mat = new Mat();
        radiusMatch_5(nativeObj, queryDescriptors.nativeObj, matches_mat.nativeObj, maxDistance);
        Converters.Mat_to_vector_vector_DMatch(matches_mat, matches);
        matches_mat.release();
    }


    //
    // C++:  void cv::DescriptorMatcher::read(FileNode arg1)
    //

    // Unknown type 'FileNode' (I), skipping the function


    //
    // C++:  void cv::DescriptorMatcher::read(String fileName)
    //

    public void read(String fileName) {
        read_0(nativeObj, fileName);
    }


    //
    // C++:  void cv::DescriptorMatcher::train()
    //

    /**
     * Trains a descriptor matcher
     *
     *     Trains a descriptor matcher (for example, the flann index). In all methods to match, the method
     *     train() is run every time before matching. Some descriptor matchers (for example, BruteForceMatcher)
     *     have an empty implementation of this method. Other matchers really train their inner structures (for
     *     example, FlannBasedMatcher trains flann::Index ).
     */
    public void train() {
        train_0(nativeObj);
    }


    //
    // C++:  void cv::DescriptorMatcher::write(Ptr_FileStorage fs, String name = String())
    //

    // Unknown type 'Ptr_FileStorage' (I), skipping the function


    //
    // C++:  void cv::DescriptorMatcher::write(String fileName)
    //

    public void write(String fileName) {
        write_0(nativeObj, fileName);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Ptr_DescriptorMatcher cv::DescriptorMatcher::clone(bool emptyTrainData = false)
    private static native long clone_0(long nativeObj, boolean emptyTrainData);
    private static native long clone_1(long nativeObj);

    // C++: static Ptr_DescriptorMatcher cv::DescriptorMatcher::create(DescriptorMatcher_MatcherType matcherType)
    private static native long create_0(int matcherType);

    // C++: static Ptr_DescriptorMatcher cv::DescriptorMatcher::create(String descriptorMatcherType)
    private static native long create_1(String descriptorMatcherType);

    // C++:  bool cv::DescriptorMatcher::empty()
    private static native boolean empty_0(long nativeObj);

    // C++:  bool cv::DescriptorMatcher::isMaskSupported()
    private static native boolean isMaskSupported_0(long nativeObj);

    // C++:  vector_Mat cv::DescriptorMatcher::getTrainDescriptors()
    private static native long getTrainDescriptors_0(long nativeObj);

    // C++:  void cv::DescriptorMatcher::add(vector_Mat descriptors)
    private static native void add_0(long nativeObj, long descriptors_mat_nativeObj);

    // C++:  void cv::DescriptorMatcher::clear()
    private static native void clear_0(long nativeObj);

    // C++:  void cv::DescriptorMatcher::knnMatch(Mat queryDescriptors, Mat trainDescriptors, vector_vector_DMatch& matches, int k, Mat mask = Mat(), bool compactResult = false)
    private static native void knnMatch_0(long nativeObj, long queryDescriptors_nativeObj, long trainDescriptors_nativeObj, long matches_mat_nativeObj, int k, long mask_nativeObj, boolean compactResult);
    private static native void knnMatch_1(long nativeObj, long queryDescriptors_nativeObj, long trainDescriptors_nativeObj, long matches_mat_nativeObj, int k, long mask_nativeObj);
    private static native void knnMatch_2(long nativeObj, long queryDescriptors_nativeObj, long trainDescriptors_nativeObj, long matches_mat_nativeObj, int k);

    // C++:  void cv::DescriptorMatcher::knnMatch(Mat queryDescriptors, vector_vector_DMatch& matches, int k, vector_Mat masks = vector_Mat(), bool compactResult = false)
    private static native void knnMatch_3(long nativeObj, long queryDescriptors_nativeObj, long matches_mat_nativeObj, int k, long masks_mat_nativeObj, boolean compactResult);
    private static native void knnMatch_4(long nativeObj, long queryDescriptors_nativeObj, long matches_mat_nativeObj, int k, long masks_mat_nativeObj);
    private static native void knnMatch_5(long nativeObj, long queryDescriptors_nativeObj, long matches_mat_nativeObj, int k);

    // C++:  void cv::DescriptorMatcher::match(Mat queryDescriptors, Mat trainDescriptors, vector_DMatch& matches, Mat mask = Mat())
    private static native void match_0(long nativeObj, long queryDescriptors_nativeObj, long trainDescriptors_nativeObj, long matches_mat_nativeObj, long mask_nativeObj);
    private static native void match_1(long nativeObj, long queryDescriptors_nativeObj, long trainDescriptors_nativeObj, long matches_mat_nativeObj);

    // C++:  void cv::DescriptorMatcher::match(Mat queryDescriptors, vector_DMatch& matches, vector_Mat masks = vector_Mat())
    private static native void match_2(long nativeObj, long queryDescriptors_nativeObj, long matches_mat_nativeObj, long masks_mat_nativeObj);
    private static native void match_3(long nativeObj, long queryDescriptors_nativeObj, long matches_mat_nativeObj);

    // C++:  void cv::DescriptorMatcher::radiusMatch(Mat queryDescriptors, Mat trainDescriptors, vector_vector_DMatch& matches, float maxDistance, Mat mask = Mat(), bool compactResult = false)
    private static native void radiusMatch_0(long nativeObj, long queryDescriptors_nativeObj, long trainDescriptors_nativeObj, long matches_mat_nativeObj, float maxDistance, long mask_nativeObj, boolean compactResult);
    private static native void radiusMatch_1(long nativeObj, long queryDescriptors_nativeObj, long trainDescriptors_nativeObj, long matches_mat_nativeObj, float maxDistance, long mask_nativeObj);
    private static native void radiusMatch_2(long nativeObj, long queryDescriptors_nativeObj, long trainDescriptors_nativeObj, long matches_mat_nativeObj, float maxDistance);

    // C++:  void cv::DescriptorMatcher::radiusMatch(Mat queryDescriptors, vector_vector_DMatch& matches, float maxDistance, vector_Mat masks = vector_Mat(), bool compactResult = false)
    private static native void radiusMatch_3(long nativeObj, long queryDescriptors_nativeObj, long matches_mat_nativeObj, float maxDistance, long masks_mat_nativeObj, boolean compactResult);
    private static native void radiusMatch_4(long nativeObj, long queryDescriptors_nativeObj, long matches_mat_nativeObj, float maxDistance, long masks_mat_nativeObj);
    private static native void radiusMatch_5(long nativeObj, long queryDescriptors_nativeObj, long matches_mat_nativeObj, float maxDistance);

    // C++:  void cv::DescriptorMatcher::read(String fileName)
    private static native void read_0(long nativeObj, String fileName);

    // C++:  void cv::DescriptorMatcher::train()
    private static native void train_0(long nativeObj);

    // C++:  void cv::DescriptorMatcher::write(String fileName)
    private static native void write_0(long nativeObj, String fileName);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
