//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.EM;
import org.opencv.ml.StatModel;
import org.opencv.utils.Converters;

// C++: class EM
/**
 * The class implements the Expectation Maximization algorithm.
 *
 * SEE: REF: ml_intro_em
 */
public class EM extends StatModel {

    protected EM(long addr) { super(addr); }

    // internal usage only
    public static EM __fromPtr__(long addr) { return new EM(addr); }

    // C++: enum <unnamed>
    public static final int
            DEFAULT_NCLUSTERS = 5,
            DEFAULT_MAX_ITERS = 100,
            START_E_STEP = 1,
            START_M_STEP = 2,
            START_AUTO_STEP = 0;


    // C++: enum Types
    public static final int
            COV_MAT_SPHERICAL = 0,
            COV_MAT_DIAGONAL = 1,
            COV_MAT_GENERIC = 2,
            COV_MAT_DEFAULT = COV_MAT_DIAGONAL;


    //
    // C++:  Mat cv::ml::EM::getMeans()
    //

    /**
     * Returns the cluster centers (means of the Gaussian mixture)
     *
     *     Returns matrix with the number of rows equal to the number of mixtures and number of columns
     *     equal to the space dimensionality.
     * @return automatically generated
     */
    public Mat getMeans() {
        return new Mat(getMeans_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::EM::getWeights()
    //

    /**
     * Returns weights of the mixtures
     *
     *     Returns vector with the number of elements equal to the number of mixtures.
     * @return automatically generated
     */
    public Mat getWeights() {
        return new Mat(getWeights_0(nativeObj));
    }


    //
    // C++: static Ptr_EM cv::ml::EM::create()
    //

    /**
     * Creates empty %EM model.
     *     The model should be trained then using StatModel::train(traindata, flags) method. Alternatively, you
     *     can use one of the EM::train\* methods or load it from file using Algorithm::load&lt;EM&gt;(filename).
     * @return automatically generated
     */
    public static EM create() {
        return EM.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_EM cv::ml::EM::load(String filepath, String nodeName = String())
    //

    /**
     * Loads and creates a serialized EM from a file
     *
     * Use EM::save to serialize and store an EM to disk.
     * Load the EM from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized EM
     * @param nodeName name of node containing the classifier
     * @return automatically generated
     */
    public static EM load(String filepath, String nodeName) {
        return EM.__fromPtr__(load_0(filepath, nodeName));
    }

    /**
     * Loads and creates a serialized EM from a file
     *
     * Use EM::save to serialize and store an EM to disk.
     * Load the EM from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized EM
     * @return automatically generated
     */
    public static EM load(String filepath) {
        return EM.__fromPtr__(load_1(filepath));
    }


    //
    // C++:  TermCriteria cv::ml::EM::getTermCriteria()
    //

    /**
     * SEE: setTermCriteria
     * @return automatically generated
     */
    public TermCriteria getTermCriteria() {
        return new TermCriteria(getTermCriteria_0(nativeObj));
    }


    //
    // C++:  Vec2d cv::ml::EM::predict2(Mat sample, Mat& probs)
    //

    /**
     * Returns a likelihood logarithm value and an index of the most probable mixture component
     *     for the given sample.
     *
     *     @param sample A sample for classification. It should be a one-channel matrix of
     *         \(1 \times dims\) or \(dims \times 1\) size.
     *     @param probs Optional output matrix that contains posterior probabilities of each component
     *         given the sample. It has \(1 \times nclusters\) size and CV_64FC1 type.
     *
     *     The method returns a two-element double vector. Zero element is a likelihood logarithm value for
     *     the sample. First element is an index of the most probable mixture component for the given
     *     sample.
     * @return automatically generated
     */
    public double[] predict2(Mat sample, Mat probs) {
        return predict2_0(nativeObj, sample.nativeObj, probs.nativeObj);
    }


    //
    // C++:  bool cv::ml::EM::trainE(Mat samples, Mat means0, Mat covs0 = Mat(), Mat weights0 = Mat(), Mat& logLikelihoods = Mat(), Mat& labels = Mat(), Mat& probs = Mat())
    //

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. You need to provide initial means \(a_k\) of
     *     mixture components. Optionally you can pass initial weights \(\pi_k\) and covariance matrices
     *     \(S_k\) of mixture components.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param means0 Initial means \(a_k\) of mixture components. It is a one-channel matrix of
     *         \(nclusters \times dims\) size. If the matrix does not have CV_64F type it will be
     *         converted to the inner matrix of such type for the further computing.
     *     @param covs0 The vector of initial covariance matrices \(S_k\) of mixture components. Each of
     *         covariance matrices is a one-channel matrix of \(dims \times dims\) size. If the matrices
     *         do not have CV_64F type they will be converted to the inner matrices of such type for the
     *         further computing.
     *     @param weights0 Initial weights \(\pi_k\) of mixture components. It should be a one-channel
     *         floating-point matrix with \(1 \times nclusters\) or \(nclusters \times 1\) size.
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *     @param labels The optional output "class label" for each sample:
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *     @param probs The optional output matrix that contains posterior probabilities of each Gaussian
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainE(Mat samples, Mat means0, Mat covs0, Mat weights0, Mat logLikelihoods, Mat labels, Mat probs) {
        return trainE_0(nativeObj, samples.nativeObj, means0.nativeObj, covs0.nativeObj, weights0.nativeObj, logLikelihoods.nativeObj, labels.nativeObj, probs.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. You need to provide initial means \(a_k\) of
     *     mixture components. Optionally you can pass initial weights \(\pi_k\) and covariance matrices
     *     \(S_k\) of mixture components.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param means0 Initial means \(a_k\) of mixture components. It is a one-channel matrix of
     *         \(nclusters \times dims\) size. If the matrix does not have CV_64F type it will be
     *         converted to the inner matrix of such type for the further computing.
     *     @param covs0 The vector of initial covariance matrices \(S_k\) of mixture components. Each of
     *         covariance matrices is a one-channel matrix of \(dims \times dims\) size. If the matrices
     *         do not have CV_64F type they will be converted to the inner matrices of such type for the
     *         further computing.
     *     @param weights0 Initial weights \(\pi_k\) of mixture components. It should be a one-channel
     *         floating-point matrix with \(1 \times nclusters\) or \(nclusters \times 1\) size.
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *     @param labels The optional output "class label" for each sample:
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainE(Mat samples, Mat means0, Mat covs0, Mat weights0, Mat logLikelihoods, Mat labels) {
        return trainE_1(nativeObj, samples.nativeObj, means0.nativeObj, covs0.nativeObj, weights0.nativeObj, logLikelihoods.nativeObj, labels.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. You need to provide initial means \(a_k\) of
     *     mixture components. Optionally you can pass initial weights \(\pi_k\) and covariance matrices
     *     \(S_k\) of mixture components.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param means0 Initial means \(a_k\) of mixture components. It is a one-channel matrix of
     *         \(nclusters \times dims\) size. If the matrix does not have CV_64F type it will be
     *         converted to the inner matrix of such type for the further computing.
     *     @param covs0 The vector of initial covariance matrices \(S_k\) of mixture components. Each of
     *         covariance matrices is a one-channel matrix of \(dims \times dims\) size. If the matrices
     *         do not have CV_64F type they will be converted to the inner matrices of such type for the
     *         further computing.
     *     @param weights0 Initial weights \(\pi_k\) of mixture components. It should be a one-channel
     *         floating-point matrix with \(1 \times nclusters\) or \(nclusters \times 1\) size.
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainE(Mat samples, Mat means0, Mat covs0, Mat weights0, Mat logLikelihoods) {
        return trainE_2(nativeObj, samples.nativeObj, means0.nativeObj, covs0.nativeObj, weights0.nativeObj, logLikelihoods.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. You need to provide initial means \(a_k\) of
     *     mixture components. Optionally you can pass initial weights \(\pi_k\) and covariance matrices
     *     \(S_k\) of mixture components.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param means0 Initial means \(a_k\) of mixture components. It is a one-channel matrix of
     *         \(nclusters \times dims\) size. If the matrix does not have CV_64F type it will be
     *         converted to the inner matrix of such type for the further computing.
     *     @param covs0 The vector of initial covariance matrices \(S_k\) of mixture components. Each of
     *         covariance matrices is a one-channel matrix of \(dims \times dims\) size. If the matrices
     *         do not have CV_64F type they will be converted to the inner matrices of such type for the
     *         further computing.
     *     @param weights0 Initial weights \(\pi_k\) of mixture components. It should be a one-channel
     *         floating-point matrix with \(1 \times nclusters\) or \(nclusters \times 1\) size.
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainE(Mat samples, Mat means0, Mat covs0, Mat weights0) {
        return trainE_3(nativeObj, samples.nativeObj, means0.nativeObj, covs0.nativeObj, weights0.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. You need to provide initial means \(a_k\) of
     *     mixture components. Optionally you can pass initial weights \(\pi_k\) and covariance matrices
     *     \(S_k\) of mixture components.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param means0 Initial means \(a_k\) of mixture components. It is a one-channel matrix of
     *         \(nclusters \times dims\) size. If the matrix does not have CV_64F type it will be
     *         converted to the inner matrix of such type for the further computing.
     *     @param covs0 The vector of initial covariance matrices \(S_k\) of mixture components. Each of
     *         covariance matrices is a one-channel matrix of \(dims \times dims\) size. If the matrices
     *         do not have CV_64F type they will be converted to the inner matrices of such type for the
     *         further computing.
     *         floating-point matrix with \(1 \times nclusters\) or \(nclusters \times 1\) size.
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainE(Mat samples, Mat means0, Mat covs0) {
        return trainE_4(nativeObj, samples.nativeObj, means0.nativeObj, covs0.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. You need to provide initial means \(a_k\) of
     *     mixture components. Optionally you can pass initial weights \(\pi_k\) and covariance matrices
     *     \(S_k\) of mixture components.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param means0 Initial means \(a_k\) of mixture components. It is a one-channel matrix of
     *         \(nclusters \times dims\) size. If the matrix does not have CV_64F type it will be
     *         converted to the inner matrix of such type for the further computing.
     *         covariance matrices is a one-channel matrix of \(dims \times dims\) size. If the matrices
     *         do not have CV_64F type they will be converted to the inner matrices of such type for the
     *         further computing.
     *         floating-point matrix with \(1 \times nclusters\) or \(nclusters \times 1\) size.
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainE(Mat samples, Mat means0) {
        return trainE_5(nativeObj, samples.nativeObj, means0.nativeObj);
    }


    //
    // C++:  bool cv::ml::EM::trainEM(Mat samples, Mat& logLikelihoods = Mat(), Mat& labels = Mat(), Mat& probs = Mat())
    //

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. Initial values of the model parameters will be
     *     estimated by the k-means algorithm.
     *
     *     Unlike many of the ML models, %EM is an unsupervised learning algorithm and it does not take
     *     responses (class labels or function values) as input. Instead, it computes the *Maximum
     *     Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the
     *     parameters inside the structure: \(p_{i,k}\) in probs, \(a_k\) in means , \(S_k\) in
     *     covs[k], \(\pi_k\) in weights , and optionally computes the output "class label" for each
     *     sample: \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most
     *     probable mixture component for each sample).
     *
     *     The trained model can be used further for prediction, just like any other classifier. The
     *     trained model is similar to the NormalBayesClassifier.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *     @param labels The optional output "class label" for each sample:
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *     @param probs The optional output matrix that contains posterior probabilities of each Gaussian
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainEM(Mat samples, Mat logLikelihoods, Mat labels, Mat probs) {
        return trainEM_0(nativeObj, samples.nativeObj, logLikelihoods.nativeObj, labels.nativeObj, probs.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. Initial values of the model parameters will be
     *     estimated by the k-means algorithm.
     *
     *     Unlike many of the ML models, %EM is an unsupervised learning algorithm and it does not take
     *     responses (class labels or function values) as input. Instead, it computes the *Maximum
     *     Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the
     *     parameters inside the structure: \(p_{i,k}\) in probs, \(a_k\) in means , \(S_k\) in
     *     covs[k], \(\pi_k\) in weights , and optionally computes the output "class label" for each
     *     sample: \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most
     *     probable mixture component for each sample).
     *
     *     The trained model can be used further for prediction, just like any other classifier. The
     *     trained model is similar to the NormalBayesClassifier.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *     @param labels The optional output "class label" for each sample:
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainEM(Mat samples, Mat logLikelihoods, Mat labels) {
        return trainEM_1(nativeObj, samples.nativeObj, logLikelihoods.nativeObj, labels.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. Initial values of the model parameters will be
     *     estimated by the k-means algorithm.
     *
     *     Unlike many of the ML models, %EM is an unsupervised learning algorithm and it does not take
     *     responses (class labels or function values) as input. Instead, it computes the *Maximum
     *     Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the
     *     parameters inside the structure: \(p_{i,k}\) in probs, \(a_k\) in means , \(S_k\) in
     *     covs[k], \(\pi_k\) in weights , and optionally computes the output "class label" for each
     *     sample: \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most
     *     probable mixture component for each sample).
     *
     *     The trained model can be used further for prediction, just like any other classifier. The
     *     trained model is similar to the NormalBayesClassifier.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainEM(Mat samples, Mat logLikelihoods) {
        return trainEM_2(nativeObj, samples.nativeObj, logLikelihoods.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Expectation step. Initial values of the model parameters will be
     *     estimated by the k-means algorithm.
     *
     *     Unlike many of the ML models, %EM is an unsupervised learning algorithm and it does not take
     *     responses (class labels or function values) as input. Instead, it computes the *Maximum
     *     Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the
     *     parameters inside the structure: \(p_{i,k}\) in probs, \(a_k\) in means , \(S_k\) in
     *     covs[k], \(\pi_k\) in weights , and optionally computes the output "class label" for each
     *     sample: \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most
     *     probable mixture component for each sample).
     *
     *     The trained model can be used further for prediction, just like any other classifier. The
     *     trained model is similar to the NormalBayesClassifier.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainEM(Mat samples) {
        return trainEM_3(nativeObj, samples.nativeObj);
    }


    //
    // C++:  bool cv::ml::EM::trainM(Mat samples, Mat probs0, Mat& logLikelihoods = Mat(), Mat& labels = Mat(), Mat& probs = Mat())
    //

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Maximization step. You need to provide initial probabilities
     *     \(p_{i,k}\) to use this option.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param probs0 the probabilities
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *     @param labels The optional output "class label" for each sample:
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *     @param probs The optional output matrix that contains posterior probabilities of each Gaussian
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainM(Mat samples, Mat probs0, Mat logLikelihoods, Mat labels, Mat probs) {
        return trainM_0(nativeObj, samples.nativeObj, probs0.nativeObj, logLikelihoods.nativeObj, labels.nativeObj, probs.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Maximization step. You need to provide initial probabilities
     *     \(p_{i,k}\) to use this option.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param probs0 the probabilities
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *     @param labels The optional output "class label" for each sample:
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainM(Mat samples, Mat probs0, Mat logLikelihoods, Mat labels) {
        return trainM_1(nativeObj, samples.nativeObj, probs0.nativeObj, logLikelihoods.nativeObj, labels.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Maximization step. You need to provide initial probabilities
     *     \(p_{i,k}\) to use this option.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param probs0 the probabilities
     *     @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainM(Mat samples, Mat probs0, Mat logLikelihoods) {
        return trainM_2(nativeObj, samples.nativeObj, probs0.nativeObj, logLikelihoods.nativeObj);
    }

    /**
     * Estimate the Gaussian mixture parameters from a samples set.
     *
     *     This variation starts with Maximization step. You need to provide initial probabilities
     *     \(p_{i,k}\) to use this option.
     *
     *     @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
     *         one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
     *         it will be converted to the inner matrix of such type for the further computing.
     *     @param probs0 the probabilities
     *         each sample. It has \(nsamples \times 1\) size and CV_64FC1 type.
     *         \(\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\) (indices of the most probable
     *         mixture component for each sample). It has \(nsamples \times 1\) size and CV_32SC1 type.
     *         mixture component given the each sample. It has \(nsamples \times nclusters\) size and
     *         CV_64FC1 type.
     * @return automatically generated
     */
    public boolean trainM(Mat samples, Mat probs0) {
        return trainM_3(nativeObj, samples.nativeObj, probs0.nativeObj);
    }


    //
    // C++:  float cv::ml::EM::predict(Mat samples, Mat& results = Mat(), int flags = 0)
    //

    /**
     * Returns posterior probabilities for the provided samples
     *
     *     @param samples The input samples, floating-point matrix
     *     @param results The optional output \( nSamples \times nClusters\) matrix of results. It contains
     *     posterior probabilities for each sample from the input
     *     @param flags This parameter will be ignored
     * @return automatically generated
     */
    public float predict(Mat samples, Mat results, int flags) {
        return predict_0(nativeObj, samples.nativeObj, results.nativeObj, flags);
    }

    /**
     * Returns posterior probabilities for the provided samples
     *
     *     @param samples The input samples, floating-point matrix
     *     @param results The optional output \( nSamples \times nClusters\) matrix of results. It contains
     *     posterior probabilities for each sample from the input
     * @return automatically generated
     */
    public float predict(Mat samples, Mat results) {
        return predict_1(nativeObj, samples.nativeObj, results.nativeObj);
    }

    /**
     * Returns posterior probabilities for the provided samples
     *
     *     @param samples The input samples, floating-point matrix
     *     posterior probabilities for each sample from the input
     * @return automatically generated
     */
    public float predict(Mat samples) {
        return predict_2(nativeObj, samples.nativeObj);
    }


    //
    // C++:  int cv::ml::EM::getClustersNumber()
    //

    /**
     * SEE: setClustersNumber
     * @return automatically generated
     */
    public int getClustersNumber() {
        return getClustersNumber_0(nativeObj);
    }


    //
    // C++:  int cv::ml::EM::getCovarianceMatrixType()
    //

    /**
     * SEE: setCovarianceMatrixType
     * @return automatically generated
     */
    public int getCovarianceMatrixType() {
        return getCovarianceMatrixType_0(nativeObj);
    }


    //
    // C++:  void cv::ml::EM::getCovs(vector_Mat& covs)
    //

    /**
     * Returns covariation matrices
     *
     *     Returns vector of covariation matrices. Number of matrices is the number of gaussian mixtures,
     *     each matrix is a square floating-point matrix NxN, where N is the space dimensionality.
     * @param covs automatically generated
     */
    public void getCovs(List<Mat> covs) {
        Mat covs_mat = new Mat();
        getCovs_0(nativeObj, covs_mat.nativeObj);
        Converters.Mat_to_vector_Mat(covs_mat, covs);
        covs_mat.release();
    }


    //
    // C++:  void cv::ml::EM::setClustersNumber(int val)
    //

    /**
     *  getClustersNumber SEE: getClustersNumber
     * @param val automatically generated
     */
    public void setClustersNumber(int val) {
        setClustersNumber_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::EM::setCovarianceMatrixType(int val)
    //

    /**
     *  getCovarianceMatrixType SEE: getCovarianceMatrixType
     * @param val automatically generated
     */
    public void setCovarianceMatrixType(int val) {
        setCovarianceMatrixType_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::EM::setTermCriteria(TermCriteria val)
    //

    /**
     *  getTermCriteria SEE: getTermCriteria
     * @param val automatically generated
     */
    public void setTermCriteria(TermCriteria val) {
        setTermCriteria_0(nativeObj, val.type, val.maxCount, val.epsilon);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::ml::EM::getMeans()
    private static native long getMeans_0(long nativeObj);

    // C++:  Mat cv::ml::EM::getWeights()
    private static native long getWeights_0(long nativeObj);

    // C++: static Ptr_EM cv::ml::EM::create()
    private static native long create_0();

    // C++: static Ptr_EM cv::ml::EM::load(String filepath, String nodeName = String())
    private static native long load_0(String filepath, String nodeName);
    private static native long load_1(String filepath);

    // C++:  TermCriteria cv::ml::EM::getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  Vec2d cv::ml::EM::predict2(Mat sample, Mat& probs)
    private static native double[] predict2_0(long nativeObj, long sample_nativeObj, long probs_nativeObj);

    // C++:  bool cv::ml::EM::trainE(Mat samples, Mat means0, Mat covs0 = Mat(), Mat weights0 = Mat(), Mat& logLikelihoods = Mat(), Mat& labels = Mat(), Mat& probs = Mat())
    private static native boolean trainE_0(long nativeObj, long samples_nativeObj, long means0_nativeObj, long covs0_nativeObj, long weights0_nativeObj, long logLikelihoods_nativeObj, long labels_nativeObj, long probs_nativeObj);
    private static native boolean trainE_1(long nativeObj, long samples_nativeObj, long means0_nativeObj, long covs0_nativeObj, long weights0_nativeObj, long logLikelihoods_nativeObj, long labels_nativeObj);
    private static native boolean trainE_2(long nativeObj, long samples_nativeObj, long means0_nativeObj, long covs0_nativeObj, long weights0_nativeObj, long logLikelihoods_nativeObj);
    private static native boolean trainE_3(long nativeObj, long samples_nativeObj, long means0_nativeObj, long covs0_nativeObj, long weights0_nativeObj);
    private static native boolean trainE_4(long nativeObj, long samples_nativeObj, long means0_nativeObj, long covs0_nativeObj);
    private static native boolean trainE_5(long nativeObj, long samples_nativeObj, long means0_nativeObj);

    // C++:  bool cv::ml::EM::trainEM(Mat samples, Mat& logLikelihoods = Mat(), Mat& labels = Mat(), Mat& probs = Mat())
    private static native boolean trainEM_0(long nativeObj, long samples_nativeObj, long logLikelihoods_nativeObj, long labels_nativeObj, long probs_nativeObj);
    private static native boolean trainEM_1(long nativeObj, long samples_nativeObj, long logLikelihoods_nativeObj, long labels_nativeObj);
    private static native boolean trainEM_2(long nativeObj, long samples_nativeObj, long logLikelihoods_nativeObj);
    private static native boolean trainEM_3(long nativeObj, long samples_nativeObj);

    // C++:  bool cv::ml::EM::trainM(Mat samples, Mat probs0, Mat& logLikelihoods = Mat(), Mat& labels = Mat(), Mat& probs = Mat())
    private static native boolean trainM_0(long nativeObj, long samples_nativeObj, long probs0_nativeObj, long logLikelihoods_nativeObj, long labels_nativeObj, long probs_nativeObj);
    private static native boolean trainM_1(long nativeObj, long samples_nativeObj, long probs0_nativeObj, long logLikelihoods_nativeObj, long labels_nativeObj);
    private static native boolean trainM_2(long nativeObj, long samples_nativeObj, long probs0_nativeObj, long logLikelihoods_nativeObj);
    private static native boolean trainM_3(long nativeObj, long samples_nativeObj, long probs0_nativeObj);

    // C++:  float cv::ml::EM::predict(Mat samples, Mat& results = Mat(), int flags = 0)
    private static native float predict_0(long nativeObj, long samples_nativeObj, long results_nativeObj, int flags);
    private static native float predict_1(long nativeObj, long samples_nativeObj, long results_nativeObj);
    private static native float predict_2(long nativeObj, long samples_nativeObj);

    // C++:  int cv::ml::EM::getClustersNumber()
    private static native int getClustersNumber_0(long nativeObj);

    // C++:  int cv::ml::EM::getCovarianceMatrixType()
    private static native int getCovarianceMatrixType_0(long nativeObj);

    // C++:  void cv::ml::EM::getCovs(vector_Mat& covs)
    private static native void getCovs_0(long nativeObj, long covs_mat_nativeObj);

    // C++:  void cv::ml::EM::setClustersNumber(int val)
    private static native void setClustersNumber_0(long nativeObj, int val);

    // C++:  void cv::ml::EM::setCovarianceMatrixType(int val)
    private static native void setCovarianceMatrixType_0(long nativeObj, int val);

    // C++:  void cv::ml::EM::setTermCriteria(TermCriteria val)
    private static native void setTermCriteria_0(long nativeObj, int val_type, int val_maxCount, double val_epsilon);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
