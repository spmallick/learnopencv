//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

// C++: class SVM
/**
 * Support Vector Machines.
 *
 * SEE: REF: ml_intro_svm
 */
public class SVM extends StatModel {

    protected SVM(long addr) { super(addr); }

    // internal usage only
    public static SVM __fromPtr__(long addr) { return new SVM(addr); }

    // C++: enum KernelTypes
    public static final int
            CUSTOM = -1,
            LINEAR = 0,
            POLY = 1,
            RBF = 2,
            SIGMOID = 3,
            CHI2 = 4,
            INTER = 5;


    // C++: enum Types
    public static final int
            C_SVC = 100,
            NU_SVC = 101,
            ONE_CLASS = 102,
            EPS_SVR = 103,
            NU_SVR = 104;


    // C++: enum ParamTypes
    public static final int
            C = 0,
            GAMMA = 1,
            P = 2,
            NU = 3,
            COEF = 4,
            DEGREE = 5;


    //
    // C++:  Mat cv::ml::SVM::getClassWeights()
    //

    /**
     * SEE: setClassWeights
     * @return automatically generated
     */
    public Mat getClassWeights() {
        return new Mat(getClassWeights_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::SVM::getSupportVectors()
    //

    /**
     * Retrieves all the support vectors
     *
     *     The method returns all the support vectors as a floating-point matrix, where support vectors are
     *     stored as matrix rows.
     * @return automatically generated
     */
    public Mat getSupportVectors() {
        return new Mat(getSupportVectors_0(nativeObj));
    }


    //
    // C++:  Mat cv::ml::SVM::getUncompressedSupportVectors()
    //

    /**
     * Retrieves all the uncompressed support vectors of a linear %SVM
     *
     *     The method returns all the uncompressed support vectors of a linear %SVM that the compressed
     *     support vector, used for prediction, was derived from. They are returned in a floating-point
     *     matrix, where the support vectors are stored as matrix rows.
     * @return automatically generated
     */
    public Mat getUncompressedSupportVectors() {
        return new Mat(getUncompressedSupportVectors_0(nativeObj));
    }


    //
    // C++: static Ptr_ParamGrid cv::ml::SVM::getDefaultGridPtr(int param_id)
    //

    /**
     * Generates a grid for %SVM parameters.
     *
     *     @param param_id %SVM parameters IDs that must be one of the SVM::ParamTypes. The grid is
     *     generated for the parameter with this ID.
     *
     *     The function generates a grid pointer for the specified parameter of the %SVM algorithm.
     *     The grid may be passed to the function SVM::trainAuto.
     * @return automatically generated
     */
    public static ParamGrid getDefaultGridPtr(int param_id) {
        return ParamGrid.__fromPtr__(getDefaultGridPtr_0(param_id));
    }


    //
    // C++: static Ptr_SVM cv::ml::SVM::create()
    //

    /**
     * Creates empty model.
     *     Use StatModel::train to train the model. Since %SVM has several parameters, you may want to
     * find the best parameters for your problem, it can be done with SVM::trainAuto.
     * @return automatically generated
     */
    public static SVM create() {
        return SVM.__fromPtr__(create_0());
    }


    //
    // C++: static Ptr_SVM cv::ml::SVM::load(String filepath)
    //

    /**
     * Loads and creates a serialized svm from a file
     *
     * Use SVM::save to serialize and store an SVM to disk.
     * Load the SVM from this file again, by calling this function with the path to the file.
     *
     * @param filepath path to serialized svm
     * @return automatically generated
     */
    public static SVM load(String filepath) {
        return SVM.__fromPtr__(load_0(filepath));
    }


    //
    // C++:  TermCriteria cv::ml::SVM::getTermCriteria()
    //

    /**
     * SEE: setTermCriteria
     * @return automatically generated
     */
    public TermCriteria getTermCriteria() {
        return new TermCriteria(getTermCriteria_0(nativeObj));
    }


    //
    // C++:  bool cv::ml::SVM::trainAuto(Mat samples, int layout, Mat responses, int kFold = 10, Ptr_ParamGrid Cgrid = SVM::getDefaultGridPtr(SVM::C), Ptr_ParamGrid gammaGrid = SVM::getDefaultGridPtr(SVM::GAMMA), Ptr_ParamGrid pGrid = SVM::getDefaultGridPtr(SVM::P), Ptr_ParamGrid nuGrid = SVM::getDefaultGridPtr(SVM::NU), Ptr_ParamGrid coeffGrid = SVM::getDefaultGridPtr(SVM::COEF), Ptr_ParamGrid degreeGrid = SVM::getDefaultGridPtr(SVM::DEGREE), bool balanced = false)
    //

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *     @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *     @param Cgrid grid for C
     *     @param gammaGrid grid for gamma
     *     @param pGrid grid for p
     *     @param nuGrid grid for nu
     *     @param coeffGrid grid for coeff
     *     @param degreeGrid grid for degree
     *     @param balanced If true and the problem is 2-class classification then the method creates more
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid, ParamGrid coeffGrid, ParamGrid degreeGrid, boolean balanced) {
        return trainAuto_0(nativeObj, samples.nativeObj, layout, responses.nativeObj, kFold, Cgrid.getNativeObjAddr(), gammaGrid.getNativeObjAddr(), pGrid.getNativeObjAddr(), nuGrid.getNativeObjAddr(), coeffGrid.getNativeObjAddr(), degreeGrid.getNativeObjAddr(), balanced);
    }

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *     @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *     @param Cgrid grid for C
     *     @param gammaGrid grid for gamma
     *     @param pGrid grid for p
     *     @param nuGrid grid for nu
     *     @param coeffGrid grid for coeff
     *     @param degreeGrid grid for degree
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid, ParamGrid coeffGrid, ParamGrid degreeGrid) {
        return trainAuto_1(nativeObj, samples.nativeObj, layout, responses.nativeObj, kFold, Cgrid.getNativeObjAddr(), gammaGrid.getNativeObjAddr(), pGrid.getNativeObjAddr(), nuGrid.getNativeObjAddr(), coeffGrid.getNativeObjAddr(), degreeGrid.getNativeObjAddr());
    }

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *     @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *     @param Cgrid grid for C
     *     @param gammaGrid grid for gamma
     *     @param pGrid grid for p
     *     @param nuGrid grid for nu
     *     @param coeffGrid grid for coeff
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid, ParamGrid coeffGrid) {
        return trainAuto_2(nativeObj, samples.nativeObj, layout, responses.nativeObj, kFold, Cgrid.getNativeObjAddr(), gammaGrid.getNativeObjAddr(), pGrid.getNativeObjAddr(), nuGrid.getNativeObjAddr(), coeffGrid.getNativeObjAddr());
    }

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *     @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *     @param Cgrid grid for C
     *     @param gammaGrid grid for gamma
     *     @param pGrid grid for p
     *     @param nuGrid grid for nu
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid) {
        return trainAuto_3(nativeObj, samples.nativeObj, layout, responses.nativeObj, kFold, Cgrid.getNativeObjAddr(), gammaGrid.getNativeObjAddr(), pGrid.getNativeObjAddr(), nuGrid.getNativeObjAddr());
    }

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *     @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *     @param Cgrid grid for C
     *     @param gammaGrid grid for gamma
     *     @param pGrid grid for p
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid) {
        return trainAuto_4(nativeObj, samples.nativeObj, layout, responses.nativeObj, kFold, Cgrid.getNativeObjAddr(), gammaGrid.getNativeObjAddr(), pGrid.getNativeObjAddr());
    }

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *     @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *     @param Cgrid grid for C
     *     @param gammaGrid grid for gamma
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid) {
        return trainAuto_5(nativeObj, samples.nativeObj, layout, responses.nativeObj, kFold, Cgrid.getNativeObjAddr(), gammaGrid.getNativeObjAddr());
    }

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *     @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *     @param Cgrid grid for C
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses, int kFold, ParamGrid Cgrid) {
        return trainAuto_6(nativeObj, samples.nativeObj, layout, responses.nativeObj, kFold, Cgrid.getNativeObjAddr());
    }

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *     @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses, int kFold) {
        return trainAuto_7(nativeObj, samples.nativeObj, layout, responses.nativeObj, kFold);
    }

    /**
     * Trains an %SVM with optimal parameters
     *
     *     @param samples training samples
     *     @param layout See ml::SampleTypes.
     *     @param responses vector of responses associated with the training samples.
     *         subset is used to test the model, the others form the train set. So, the %SVM algorithm is
     *         balanced cross-validation subsets that is proportions between classes in subsets are close
     *         to such proportion in the whole train dataset.
     *
     *     The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
     *     nu, coef0, degree. Parameters are considered optimal when the cross-validation
     *     estimate of the test set error is minimal.
     *
     *     This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
     *     offers rudimentary parameter options.
     *
     *     This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
     *     regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
     *     the usual %SVM with parameters specified in params is executed.
     * @return automatically generated
     */
    public boolean trainAuto(Mat samples, int layout, Mat responses) {
        return trainAuto_8(nativeObj, samples.nativeObj, layout, responses.nativeObj);
    }


    //
    // C++:  double cv::ml::SVM::getC()
    //

    /**
     * SEE: setC
     * @return automatically generated
     */
    public double getC() {
        return getC_0(nativeObj);
    }


    //
    // C++:  double cv::ml::SVM::getCoef0()
    //

    /**
     * SEE: setCoef0
     * @return automatically generated
     */
    public double getCoef0() {
        return getCoef0_0(nativeObj);
    }


    //
    // C++:  double cv::ml::SVM::getDecisionFunction(int i, Mat& alpha, Mat& svidx)
    //

    /**
     * Retrieves the decision function
     *
     *     @param i the index of the decision function. If the problem solved is regression, 1-class or
     *         2-class classification, then there will be just one decision function and the index should
     *         always be 0. Otherwise, in the case of N-class classification, there will be \(N(N-1)/2\)
     *         decision functions.
     *     @param alpha the optional output vector for weights, corresponding to different support vectors.
     *         In the case of linear %SVM all the alpha's will be 1's.
     *     @param svidx the optional output vector of indices of support vectors within the matrix of
     *         support vectors (which can be retrieved by SVM::getSupportVectors). In the case of linear
     *         %SVM each decision function consists of a single "compressed" support vector.
     *
     *     The method returns rho parameter of the decision function, a scalar subtracted from the weighted
     *     sum of kernel responses.
     * @return automatically generated
     */
    public double getDecisionFunction(int i, Mat alpha, Mat svidx) {
        return getDecisionFunction_0(nativeObj, i, alpha.nativeObj, svidx.nativeObj);
    }


    //
    // C++:  double cv::ml::SVM::getDegree()
    //

    /**
     * SEE: setDegree
     * @return automatically generated
     */
    public double getDegree() {
        return getDegree_0(nativeObj);
    }


    //
    // C++:  double cv::ml::SVM::getGamma()
    //

    /**
     * SEE: setGamma
     * @return automatically generated
     */
    public double getGamma() {
        return getGamma_0(nativeObj);
    }


    //
    // C++:  double cv::ml::SVM::getNu()
    //

    /**
     * SEE: setNu
     * @return automatically generated
     */
    public double getNu() {
        return getNu_0(nativeObj);
    }


    //
    // C++:  double cv::ml::SVM::getP()
    //

    /**
     * SEE: setP
     * @return automatically generated
     */
    public double getP() {
        return getP_0(nativeObj);
    }


    //
    // C++:  int cv::ml::SVM::getKernelType()
    //

    /**
     * Type of a %SVM kernel.
     * See SVM::KernelTypes. Default value is SVM::RBF.
     * @return automatically generated
     */
    public int getKernelType() {
        return getKernelType_0(nativeObj);
    }


    //
    // C++:  int cv::ml::SVM::getType()
    //

    /**
     * SEE: setType
     * @return automatically generated
     */
    public int getType() {
        return getType_0(nativeObj);
    }


    //
    // C++:  void cv::ml::SVM::setC(double val)
    //

    /**
     *  getC SEE: getC
     * @param val automatically generated
     */
    public void setC(double val) {
        setC_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::SVM::setClassWeights(Mat val)
    //

    /**
     *  getClassWeights SEE: getClassWeights
     * @param val automatically generated
     */
    public void setClassWeights(Mat val) {
        setClassWeights_0(nativeObj, val.nativeObj);
    }


    //
    // C++:  void cv::ml::SVM::setCoef0(double val)
    //

    /**
     *  getCoef0 SEE: getCoef0
     * @param val automatically generated
     */
    public void setCoef0(double val) {
        setCoef0_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::SVM::setDegree(double val)
    //

    /**
     *  getDegree SEE: getDegree
     * @param val automatically generated
     */
    public void setDegree(double val) {
        setDegree_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::SVM::setGamma(double val)
    //

    /**
     *  getGamma SEE: getGamma
     * @param val automatically generated
     */
    public void setGamma(double val) {
        setGamma_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::SVM::setKernel(int kernelType)
    //

    /**
     * Initialize with one of predefined kernels.
     * See SVM::KernelTypes.
     * @param kernelType automatically generated
     */
    public void setKernel(int kernelType) {
        setKernel_0(nativeObj, kernelType);
    }


    //
    // C++:  void cv::ml::SVM::setNu(double val)
    //

    /**
     *  getNu SEE: getNu
     * @param val automatically generated
     */
    public void setNu(double val) {
        setNu_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::SVM::setP(double val)
    //

    /**
     *  getP SEE: getP
     * @param val automatically generated
     */
    public void setP(double val) {
        setP_0(nativeObj, val);
    }


    //
    // C++:  void cv::ml::SVM::setTermCriteria(TermCriteria val)
    //

    /**
     *  getTermCriteria SEE: getTermCriteria
     * @param val automatically generated
     */
    public void setTermCriteria(TermCriteria val) {
        setTermCriteria_0(nativeObj, val.type, val.maxCount, val.epsilon);
    }


    //
    // C++:  void cv::ml::SVM::setType(int val)
    //

    /**
     *  getType SEE: getType
     * @param val automatically generated
     */
    public void setType(int val) {
        setType_0(nativeObj, val);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat cv::ml::SVM::getClassWeights()
    private static native long getClassWeights_0(long nativeObj);

    // C++:  Mat cv::ml::SVM::getSupportVectors()
    private static native long getSupportVectors_0(long nativeObj);

    // C++:  Mat cv::ml::SVM::getUncompressedSupportVectors()
    private static native long getUncompressedSupportVectors_0(long nativeObj);

    // C++: static Ptr_ParamGrid cv::ml::SVM::getDefaultGridPtr(int param_id)
    private static native long getDefaultGridPtr_0(int param_id);

    // C++: static Ptr_SVM cv::ml::SVM::create()
    private static native long create_0();

    // C++: static Ptr_SVM cv::ml::SVM::load(String filepath)
    private static native long load_0(String filepath);

    // C++:  TermCriteria cv::ml::SVM::getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  bool cv::ml::SVM::trainAuto(Mat samples, int layout, Mat responses, int kFold = 10, Ptr_ParamGrid Cgrid = SVM::getDefaultGridPtr(SVM::C), Ptr_ParamGrid gammaGrid = SVM::getDefaultGridPtr(SVM::GAMMA), Ptr_ParamGrid pGrid = SVM::getDefaultGridPtr(SVM::P), Ptr_ParamGrid nuGrid = SVM::getDefaultGridPtr(SVM::NU), Ptr_ParamGrid coeffGrid = SVM::getDefaultGridPtr(SVM::COEF), Ptr_ParamGrid degreeGrid = SVM::getDefaultGridPtr(SVM::DEGREE), bool balanced = false)
    private static native boolean trainAuto_0(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj, int kFold, long Cgrid_nativeObj, long gammaGrid_nativeObj, long pGrid_nativeObj, long nuGrid_nativeObj, long coeffGrid_nativeObj, long degreeGrid_nativeObj, boolean balanced);
    private static native boolean trainAuto_1(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj, int kFold, long Cgrid_nativeObj, long gammaGrid_nativeObj, long pGrid_nativeObj, long nuGrid_nativeObj, long coeffGrid_nativeObj, long degreeGrid_nativeObj);
    private static native boolean trainAuto_2(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj, int kFold, long Cgrid_nativeObj, long gammaGrid_nativeObj, long pGrid_nativeObj, long nuGrid_nativeObj, long coeffGrid_nativeObj);
    private static native boolean trainAuto_3(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj, int kFold, long Cgrid_nativeObj, long gammaGrid_nativeObj, long pGrid_nativeObj, long nuGrid_nativeObj);
    private static native boolean trainAuto_4(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj, int kFold, long Cgrid_nativeObj, long gammaGrid_nativeObj, long pGrid_nativeObj);
    private static native boolean trainAuto_5(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj, int kFold, long Cgrid_nativeObj, long gammaGrid_nativeObj);
    private static native boolean trainAuto_6(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj, int kFold, long Cgrid_nativeObj);
    private static native boolean trainAuto_7(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj, int kFold);
    private static native boolean trainAuto_8(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj);

    // C++:  double cv::ml::SVM::getC()
    private static native double getC_0(long nativeObj);

    // C++:  double cv::ml::SVM::getCoef0()
    private static native double getCoef0_0(long nativeObj);

    // C++:  double cv::ml::SVM::getDecisionFunction(int i, Mat& alpha, Mat& svidx)
    private static native double getDecisionFunction_0(long nativeObj, int i, long alpha_nativeObj, long svidx_nativeObj);

    // C++:  double cv::ml::SVM::getDegree()
    private static native double getDegree_0(long nativeObj);

    // C++:  double cv::ml::SVM::getGamma()
    private static native double getGamma_0(long nativeObj);

    // C++:  double cv::ml::SVM::getNu()
    private static native double getNu_0(long nativeObj);

    // C++:  double cv::ml::SVM::getP()
    private static native double getP_0(long nativeObj);

    // C++:  int cv::ml::SVM::getKernelType()
    private static native int getKernelType_0(long nativeObj);

    // C++:  int cv::ml::SVM::getType()
    private static native int getType_0(long nativeObj);

    // C++:  void cv::ml::SVM::setC(double val)
    private static native void setC_0(long nativeObj, double val);

    // C++:  void cv::ml::SVM::setClassWeights(Mat val)
    private static native void setClassWeights_0(long nativeObj, long val_nativeObj);

    // C++:  void cv::ml::SVM::setCoef0(double val)
    private static native void setCoef0_0(long nativeObj, double val);

    // C++:  void cv::ml::SVM::setDegree(double val)
    private static native void setDegree_0(long nativeObj, double val);

    // C++:  void cv::ml::SVM::setGamma(double val)
    private static native void setGamma_0(long nativeObj, double val);

    // C++:  void cv::ml::SVM::setKernel(int kernelType)
    private static native void setKernel_0(long nativeObj, int kernelType);

    // C++:  void cv::ml::SVM::setNu(double val)
    private static native void setNu_0(long nativeObj, double val);

    // C++:  void cv::ml::SVM::setP(double val)
    private static native void setP_0(long nativeObj, double val);

    // C++:  void cv::ml::SVM::setTermCriteria(TermCriteria val)
    private static native void setTermCriteria_0(long nativeObj, int val_type, int val_maxCount, double val_epsilon);

    // C++:  void cv::ml::SVM::setType(int val)
    private static native void setType_0(long nativeObj, int val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
