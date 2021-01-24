//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.ml.TrainData;
import org.opencv.utils.Converters;

// C++: class TrainData
/**
 * Class encapsulating training data.
 *
 * Please note that the class only specifies the interface of training data, but not implementation.
 * All the statistical model classes in _ml_ module accepts Ptr&lt;TrainData&gt; as parameter. In other
 * words, you can create your own class derived from TrainData and pass smart pointer to the instance
 * of this class into StatModel::train.
 *
 * SEE: REF: ml_intro_data
 */
public class TrainData {

    protected final long nativeObj;
    protected TrainData(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static TrainData __fromPtr__(long addr) { return new TrainData(addr); }

    //
    // C++:  Mat cv::ml::TrainData::getCatMap()
    //

    public Mat getCatMap() {
        return new Mat(getCatMap_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getCatOfs()
    //

    public Mat getCatOfs() {
        return new Mat(getCatOfs_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getClassLabels()
    //

    /**
     * Returns the vector of class labels
     *
     *     The function returns vector of unique labels occurred in the responses.
     * @return automatically generated
     */
    public Mat getClassLabels() {
        return new Mat(getClassLabels_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getDefaultSubstValues()
    //

    public Mat getDefaultSubstValues() {
        return new Mat(getDefaultSubstValues_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getMissing()
    //

    public Mat getMissing() {
        return new Mat(getMissing_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getNormCatResponses()
    //

    public Mat getNormCatResponses() {
        return new Mat(getNormCatResponses_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getResponses()
    //

    public Mat getResponses() {
        return new Mat(getResponses_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getSampleWeights()
    //

    public Mat getSampleWeights() {
        return new Mat(getSampleWeights_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getSamples()
    //

    public Mat getSamples() {
        return new Mat(getSamples_0(nativeObj));
    }


    //
    // C++: static Mat cv::ml::TrainData::getSubMatrix(Mat matrix, Mat idx, int layout)
    //

    /**
     * Extract from matrix rows/cols specified by passed indexes.
     *     @param matrix input matrix (supported types: CV_32S, CV_32F, CV_64F)
     *     @param idx 1D index vector
     *     @param layout specifies to extract rows (cv::ml::ROW_SAMPLES) or to extract columns (cv::ml::COL_SAMPLES)
     * @return automatically generated
     */
    public static Mat getSubMatrix(Mat matrix, Mat idx, int layout) {
        return new Mat(getSubMatrix_0(matrix.nativeObj, idx.nativeObj, layout));
    }


    //
    // C++: static Mat cv::ml::TrainData::getSubVector(Mat vec, Mat idx)
    //

    /**
     * Extract from 1D vector elements specified by passed indexes.
     *     @param vec input vector (supported types: CV_32S, CV_32F, CV_64F)
     *     @param idx 1D index vector
     * @return automatically generated
     */
    public static Mat getSubVector(Mat vec, Mat idx) {
        return new Mat(getSubVector_0(vec.nativeObj, idx.nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTestNormCatResponses()
    //

    public Mat getTestNormCatResponses() {
        return new Mat(getTestNormCatResponses_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTestResponses()
    //

    public Mat getTestResponses() {
        return new Mat(getTestResponses_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTestSampleIdx()
    //

    public Mat getTestSampleIdx() {
        return new Mat(getTestSampleIdx_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTestSampleWeights()
    //

    public Mat getTestSampleWeights() {
        return new Mat(getTestSampleWeights_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTestSamples()
    //

    /**
     * Returns matrix of test samples
     * @return automatically generated
     */
    public Mat getTestSamples() {
        return new Mat(getTestSamples_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTrainNormCatResponses()
    //

    /**
     * Returns the vector of normalized categorical responses
     *
     *     The function returns vector of responses. Each response is integer from {@code 0} to `&lt;number of
     *     classes&gt;-1`. The actual label value can be retrieved then from the class label vector, see
     *     TrainData::getClassLabels.
     * @return automatically generated
     */
    public Mat getTrainNormCatResponses() {
        return new Mat(getTrainNormCatResponses_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTrainResponses()
    //

    /**
     * Returns the vector of responses
     *
     *     The function returns ordered or the original categorical responses. Usually it's used in
     *     regression algorithms.
     * @return automatically generated
     */
    public Mat getTrainResponses() {
        return new Mat(getTrainResponses_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTrainSampleIdx()
    //

    public Mat getTrainSampleIdx() {
        return new Mat(getTrainSampleIdx_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTrainSampleWeights()
    //

    public Mat getTrainSampleWeights() {
        return new Mat(getTrainSampleWeights_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getTrainSamples(int layout = ROW_SAMPLE, bool compressSamples = true, bool compressVars = true)
    //

    /**
     * Returns matrix of train samples
     *
     *     @param layout The requested layout. If it's different from the initial one, the matrix is
     *         transposed. See ml::SampleTypes.
     *     @param compressSamples if true, the function returns only the training samples (specified by
     *         sampleIdx)
     *     @param compressVars if true, the function returns the shorter training samples, containing only
     *         the active variables.
     *
     *     In current implementation the function tries to avoid physical data copying and returns the
     *     matrix stored inside TrainData (unless the transposition or compression is needed).
     * @return automatically generated
     */
    public Mat getTrainSamples(int layout, boolean compressSamples, boolean compressVars) {
        return new Mat(getTrainSamples_0(nativeObj, layout, compressSamples, compressVars));
    }

    /**
     * Returns matrix of train samples
     *
     *     @param layout The requested layout. If it's different from the initial one, the matrix is
     *         transposed. See ml::SampleTypes.
     *     @param compressSamples if true, the function returns only the training samples (specified by
     *         sampleIdx)
     *         the active variables.
     *
     *     In current implementation the function tries to avoid physical data copying and returns the
     *     matrix stored inside TrainData (unless the transposition or compression is needed).
     * @return automatically generated
     */
    public Mat getTrainSamples(int layout, boolean compressSamples) {
        return new Mat(getTrainSamples_1(nativeObj, layout, compressSamples));
    }

    /**
     * Returns matrix of train samples
     *
     *     @param layout The requested layout. If it's different from the initial one, the matrix is
     *         transposed. See ml::SampleTypes.
     *         sampleIdx)
     *         the active variables.
     *
     *     In current implementation the function tries to avoid physical data copying and returns the
     *     matrix stored inside TrainData (unless the transposition or compression is needed).
     * @return automatically generated
     */
    public Mat getTrainSamples(int layout) {
        return new Mat(getTrainSamples_2(nativeObj, layout));
    }

    /**
     * Returns matrix of train samples
     *
     *         transposed. See ml::SampleTypes.
     *         sampleIdx)
     *         the active variables.
     *
     *     In current implementation the function tries to avoid physical data copying and returns the
     *     matrix stored inside TrainData (unless the transposition or compression is needed).
     * @return automatically generated
     */
    public Mat getTrainSamples() {
        return new Mat(getTrainSamples_3(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getVarIdx()
    //

    public Mat getVarIdx() {
        return new Mat(getVarIdx_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getVarSymbolFlags()
    //

    public Mat getVarSymbolFlags() {
        return new Mat(getVarSymbolFlags_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::TrainData::getVarType()
    //

    public Mat getVarType() {
        return new Mat(getVarType_0(nativeObj));
    }


    //
    // C++: static Ptr_TrainData cv::ml::TrainData::create(Mat samples, int layout, Mat responses, Mat varIdx = Mat(), Mat sampleIdx = Mat(), Mat sampleWeights = Mat(), Mat varType = Mat())
    //

    /**
     * Creates training data from in-memory arrays.
     *
     *     @param samples matrix of samples. It should have CV_32F type.
     *     @param layout see ml::SampleTypes.
     *     @param responses matrix of responses. If the responses are scalar, they should be stored as a
     *         single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
     *         former case the responses are considered as ordered by default; in the latter case - as
     *         categorical)
     *     @param varIdx vector specifying which variables to use for training. It can be an integer vector
     *         (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
     *         active variables.
     *     @param sampleIdx vector specifying which samples to use for training. It can be an integer
     *         vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
     *         of training samples.
     *     @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
     *     @param varType optional vector of type CV_8U and size `&lt;number_of_variables_in_samples&gt; +
     *         &lt;number_of_variables_in_responses&gt;`, containing types of each input and output variable. See
     *         ml::VariableTypes.
     * @return automatically generated
     */
    public static TrainData create(Mat samples, int layout, Mat responses, Mat varIdx, Mat sampleIdx, Mat sampleWeights, Mat varType) {
        return TrainData.__fromPtr__(create_0(samples.nativeObj, layout, responses.nativeObj, varIdx.nativeObj, sampleIdx.nativeObj, sampleWeights.nativeObj, varType.nativeObj));
    }

    /**
     * Creates training data from in-memory arrays.
     *
     *     @param samples matrix of samples. It should have CV_32F type.
     *     @param layout see ml::SampleTypes.
     *     @param responses matrix of responses. If the responses are scalar, they should be stored as a
     *         single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
     *         former case the responses are considered as ordered by default; in the latter case - as
     *         categorical)
     *     @param varIdx vector specifying which variables to use for training. It can be an integer vector
     *         (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
     *         active variables.
     *     @param sampleIdx vector specifying which samples to use for training. It can be an integer
     *         vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
     *         of training samples.
     *     @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
     *         &lt;number_of_variables_in_responses&gt;`, containing types of each input and output variable. See
     *         ml::VariableTypes.
     * @return automatically generated
     */
    public static TrainData create(Mat samples, int layout, Mat responses, Mat varIdx, Mat sampleIdx, Mat sampleWeights) {
        return TrainData.__fromPtr__(create_1(samples.nativeObj, layout, responses.nativeObj, varIdx.nativeObj, sampleIdx.nativeObj, sampleWeights.nativeObj));
    }

    /**
     * Creates training data from in-memory arrays.
     *
     *     @param samples matrix of samples. It should have CV_32F type.
     *     @param layout see ml::SampleTypes.
     *     @param responses matrix of responses. If the responses are scalar, they should be stored as a
     *         single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
     *         former case the responses are considered as ordered by default; in the latter case - as
     *         categorical)
     *     @param varIdx vector specifying which variables to use for training. It can be an integer vector
     *         (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
     *         active variables.
     *     @param sampleIdx vector specifying which samples to use for training. It can be an integer
     *         vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
     *         of training samples.
     *         &lt;number_of_variables_in_responses&gt;`, containing types of each input and output variable. See
     *         ml::VariableTypes.
     * @return automatically generated
     */
    public static TrainData create(Mat samples, int layout, Mat responses, Mat varIdx, Mat sampleIdx) {
        return TrainData.__fromPtr__(create_2(samples.nativeObj, layout, responses.nativeObj, varIdx.nativeObj, sampleIdx.nativeObj));
    }

    /**
     * Creates training data from in-memory arrays.
     *
     *     @param samples matrix of samples. It should have CV_32F type.
     *     @param layout see ml::SampleTypes.
     *     @param responses matrix of responses. If the responses are scalar, they should be stored as a
     *         single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
     *         former case the responses are considered as ordered by default; in the latter case - as
     *         categorical)
     *     @param varIdx vector specifying which variables to use for training. It can be an integer vector
     *         (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
     *         active variables.
     *         vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
     *         of training samples.
     *         &lt;number_of_variables_in_responses&gt;`, containing types of each input and output variable. See
     *         ml::VariableTypes.
     * @return automatically generated
     */
    public static TrainData create(Mat samples, int layout, Mat responses, Mat varIdx) {
        return TrainData.__fromPtr__(create_3(samples.nativeObj, layout, responses.nativeObj, varIdx.nativeObj));
    }

    /**
     * Creates training data from in-memory arrays.
     *
     *     @param samples matrix of samples. It should have CV_32F type.
     *     @param layout see ml::SampleTypes.
     *     @param responses matrix of responses. If the responses are scalar, they should be stored as a
     *         single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
     *         former case the responses are considered as ordered by default; in the latter case - as
     *         categorical)
     *         (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
     *         active variables.
     *         vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
     *         of training samples.
     *         &lt;number_of_variables_in_responses&gt;`, containing types of each input and output variable. See
     *         ml::VariableTypes.
     * @return automatically generated
     */
    public static TrainData create(Mat samples, int layout, Mat responses) {
        return TrainData.__fromPtr__(create_4(samples.nativeObj, layout, responses.nativeObj));
    }


    //
    // C++:  int cv::ml::TrainData::getCatCount(int vi)
    //

    public int getCatCount(int vi) {
        return getCatCount_0(nativeObj, vi);
    }


    //
    // C++:  int cv::ml::TrainData::getLayout()
    //

    public int getLayout() {
        return getLayout_0(nativeObj);
    }


    //
    // C++:  int cv::ml::TrainData::getNAllVars()
    //

    public int getNAllVars() {
        return getNAllVars_0(nativeObj);
    }


    //
    // C++:  int cv::ml::TrainData::getNSamples()
    //

    public int getNSamples() {
        return getNSamples_0(nativeObj);
    }


    //
    // C++:  int cv::ml::TrainData::getNTestSamples()
    //

    public int getNTestSamples() {
        return getNTestSamples_0(nativeObj);
    }


    //
    // C++:  int cv::ml::TrainData::getNTrainSamples()
    //

    public int getNTrainSamples() {
        return getNTrainSamples_0(nativeObj);
    }


    //
    // C++:  int cv::ml::TrainData::getNVars()
    //

    public int getNVars() {
        return getNVars_0(nativeObj);
    }


    //
    // C++:  int cv::ml::TrainData::getResponseType()
    //

    public int getResponseType() {
        return getResponseType_0(nativeObj);
    }


    //
    // C++:  void cv::ml::TrainData::getNames(vector_String names)
    //

    /**
     * Returns vector of symbolic names captured in loadFromCSV()
     * @param names automatically generated
     */
    public void getNames(List<String> names) {
        getNames_0(nativeObj, names);
    }


    //
    // C++:  void cv::ml::TrainData::getSample(Mat varIdx, int sidx, float* buf)
    //

    public void getSample(Mat varIdx, int sidx, float buf) {
        getSample_0(nativeObj, varIdx.nativeObj, sidx, buf);
    }


    //
    // C++:  void cv::ml::TrainData::getValues(int vi, Mat sidx, float* values)
    //

    public void getValues(int vi, Mat sidx, float values) {
        getValues_0(nativeObj, vi, sidx.nativeObj, values);
    }


    //
    // C++:  void cv::ml::TrainData::setTrainTestSplit(int count, bool shuffle = true)
    //

    /**
     * Splits the training data into the training and test parts
     *     SEE: TrainData::setTrainTestSplitRatio
     * @param count automatically generated
     * @param shuffle automatically generated
     */
    public void setTrainTestSplit(int count, boolean shuffle) {
        setTrainTestSplit_0(nativeObj, count, shuffle);
    }

    /**
     * Splits the training data into the training and test parts
     *     SEE: TrainData::setTrainTestSplitRatio
     * @param count automatically generated
     */
    public void setTrainTestSplit(int count) {
        setTrainTestSplit_1(nativeObj, count);
    }


    //
    // C++:  void cv::ml::TrainData::setTrainTestSplitRatio(double ratio, bool shuffle = true)
    //

    /**
     * Splits the training data into the training and test parts
     *
     *     The function selects a subset of specified relative size and then returns it as the training
     *     set. If the function is not called, all the data is used for training. Please, note that for
     *     each of TrainData::getTrain\* there is corresponding TrainData::getTest\*, so that the test
     *     subset can be retrieved and processed as well.
     *     SEE: TrainData::setTrainTestSplit
     * @param ratio automatically generated
     * @param shuffle automatically generated
     */
    public void setTrainTestSplitRatio(double ratio, boolean shuffle) {
        setTrainTestSplitRatio_0(nativeObj, ratio, shuffle);
    }

    /**
     * Splits the training data into the training and test parts
     *
     *     The function selects a subset of specified relative size and then returns it as the training
     *     set. If the function is not called, all the data is used for training. Please, note that for
     *     each of TrainData::getTrain\* there is corresponding TrainData::getTest\*, so that the test
     *     subset can be retrieved and processed as well.
     *     SEE: TrainData::setTrainTestSplit
     * @param ratio automatically generated
     */
    public void setTrainTestSplitRatio(double ratio) {
        setTrainTestSplitRatio_1(nativeObj, ratio);
    }


    //
    // C++:  void cv::ml::TrainData::shuffleTrainTest()
    //

    public void shuffleTrainTest() {
        shuffleTrainTest_0(nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::ml::TrainData::getCatMap()
    private static native long getCatMap_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getCatOfs()
    private static native long getCatOfs_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getClassLabels()
    private static native long getClassLabels_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getDefaultSubstValues()
    private static native long getDefaultSubstValues_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getMissing()
    private static native long getMissing_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getNormCatResponses()
    private static native long getNormCatResponses_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getResponses()
    private static native long getResponses_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getSampleWeights()
    private static native long getSampleWeights_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getSamples()
    private static native long getSamples_0(long nativeObj);

    // C++: static Mat cv::ml::TrainData::getSubMatrix(Mat matrix, Mat idx, int layout)
    private static native long getSubMatrix_0(long matrix_nativeObj, long idx_nativeObj, int layout);

    // C++: static Mat cv::ml::TrainData::getSubVector(Mat vec, Mat idx)
    private static native long getSubVector_0(long vec_nativeObj, long idx_nativeObj);

    // C++:  Mat cv::ml::TrainData::getTestNormCatResponses()
    private static native long getTestNormCatResponses_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTestResponses()
    private static native long getTestResponses_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTestSampleIdx()
    private static native long getTestSampleIdx_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTestSampleWeights()
    private static native long getTestSampleWeights_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTestSamples()
    private static native long getTestSamples_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTrainNormCatResponses()
    private static native long getTrainNormCatResponses_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTrainResponses()
    private static native long getTrainResponses_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTrainSampleIdx()
    private static native long getTrainSampleIdx_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTrainSampleWeights()
    private static native long getTrainSampleWeights_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getTrainSamples(int layout = ROW_SAMPLE, bool compressSamples = true, bool compressVars = true)
    private static native long getTrainSamples_0(long nativeObj, int layout, boolean compressSamples, boolean compressVars);
    private static native long getTrainSamples_1(long nativeObj, int layout, boolean compressSamples);
    private static native long getTrainSamples_2(long nativeObj, int layout);
    private static native long getTrainSamples_3(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getVarIdx()
    private static native long getVarIdx_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getVarSymbolFlags()
    private static native long getVarSymbolFlags_0(long nativeObj);

    // C++:  Mat cv::ml::TrainData::getVarType()
    private static native long getVarType_0(long nativeObj);

    // C++: static Ptr_TrainData cv::ml::TrainData::create(Mat samples, int layout, Mat responses, Mat varIdx = Mat(), Mat sampleIdx = Mat(), Mat sampleWeights = Mat(), Mat varType = Mat())
    private static native long create_0(long samples_nativeObj, int layout, long responses_nativeObj, long varIdx_nativeObj, long sampleIdx_nativeObj, long sampleWeights_nativeObj, long varType_nativeObj);
    private static native long create_1(long samples_nativeObj, int layout, long responses_nativeObj, long varIdx_nativeObj, long sampleIdx_nativeObj, long sampleWeights_nativeObj);
    private static native long create_2(long samples_nativeObj, int layout, long responses_nativeObj, long varIdx_nativeObj, long sampleIdx_nativeObj);
    private static native long create_3(long samples_nativeObj, int layout, long responses_nativeObj, long varIdx_nativeObj);
    private static native long create_4(long samples_nativeObj, int layout, long responses_nativeObj);

    // C++:  int cv::ml::TrainData::getCatCount(int vi)
    private static native int getCatCount_0(long nativeObj, int vi);

    // C++:  int cv::ml::TrainData::getLayout()
    private static native int getLayout_0(long nativeObj);

    // C++:  int cv::ml::TrainData::getNAllVars()
    private static native int getNAllVars_0(long nativeObj);

    // C++:  int cv::ml::TrainData::getNSamples()
    private static native int getNSamples_0(long nativeObj);

    // C++:  int cv::ml::TrainData::getNTestSamples()
    private static native int getNTestSamples_0(long nativeObj);

    // C++:  int cv::ml::TrainData::getNTrainSamples()
    private static native int getNTrainSamples_0(long nativeObj);

    // C++:  int cv::ml::TrainData::getNVars()
    private static native int getNVars_0(long nativeObj);

    // C++:  int cv::ml::TrainData::getResponseType()
    private static native int getResponseType_0(long nativeObj);

    // C++:  void cv::ml::TrainData::getNames(vector_String names)
    private static native void getNames_0(long nativeObj, List<String> names);

    // C++:  void cv::ml::TrainData::getSample(Mat varIdx, int sidx, float* buf)
    private static native void getSample_0(long nativeObj, long varIdx_nativeObj, int sidx, float buf);

    // C++:  void cv::ml::TrainData::getValues(int vi, Mat sidx, float* values)
    private static native void getValues_0(long nativeObj, int vi, long sidx_nativeObj, float values);

    // C++:  void cv::ml::TrainData::setTrainTestSplit(int count, bool shuffle = true)
    private static native void setTrainTestSplit_0(long nativeObj, int count, boolean shuffle);
    private static native void setTrainTestSplit_1(long nativeObj, int count);

    // C++:  void cv::ml::TrainData::setTrainTestSplitRatio(double ratio, bool shuffle = true)
    private static native void setTrainTestSplitRatio_0(long nativeObj, double ratio, boolean shuffle);
    private static native void setTrainTestSplitRatio_1(long nativeObj, double ratio);

    // C++:  void cv::ml::TrainData::shuffleTrainTest()
    private static native void shuffleTrainTest_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
