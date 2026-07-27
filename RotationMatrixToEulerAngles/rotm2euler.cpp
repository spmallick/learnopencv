/*
 * Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
 * All rights reserved. No warranty, explicit or implicit, provided.
 */

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kDefaultTolerance = 1e-9;

cv::Mat asDoubleRotationMatrix(const cv::Mat& input) {
    if (input.rows != 3 || input.cols != 3 || input.channels() != 1) {
        throw std::invalid_argument("rotation matrix must be single-channel and 3x3");
    }
    if (input.depth() != CV_32F && input.depth() != CV_64F) {
        throw std::invalid_argument("rotation matrix must use float32 or float64 values");
    }

    cv::Mat rotation;
    input.convertTo(rotation, CV_64F);
    for (int row = 0; row < rotation.rows; ++row) {
        for (int column = 0; column < rotation.cols; ++column) {
            if (!std::isfinite(rotation.at<double>(row, column))) {
                throw std::invalid_argument("rotation matrix must contain finite values");
            }
        }
    }
    return rotation;
}

bool isRotationMatrix(const cv::Mat& input, const double tolerance = kDefaultTolerance) {
    if (!(tolerance > 0.0)) {
        throw std::invalid_argument("tolerance must be positive");
    }

    cv::Mat rotation;
    try {
        rotation = asDoubleRotationMatrix(input);
    } catch (const std::invalid_argument&) {
        return false;
    }

    const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat shouldBeIdentity = rotation.t() * rotation;
    const double identityError = cv::norm(identity - shouldBeIdentity, cv::NORM_INF);
    const double determinantError = std::abs(cv::determinant(rotation) - 1.0);
    return identityError <= tolerance && determinantError <= tolerance;
}

cv::Mat eulerAnglesToRotationMatrix(const cv::Vec3d& theta) {
    for (int index = 0; index < 3; ++index) {
        if (!std::isfinite(theta[index])) {
            throw std::invalid_argument("Euler angles must contain finite values");
        }
    }

    const double sinX = std::sin(theta[0]);
    const double cosX = std::cos(theta[0]);
    const double sinY = std::sin(theta[1]);
    const double cosY = std::cos(theta[1]);
    const double sinZ = std::sin(theta[2]);
    const double cosZ = std::cos(theta[2]);

    const cv::Mat rotationX = cv::Mat(cv::Matx33d(
        1.0, 0.0, 0.0,
        0.0, cosX, -sinX,
        0.0, sinX, cosX));
    const cv::Mat rotationY = cv::Mat(cv::Matx33d(
        cosY, 0.0, sinY,
        0.0, 1.0, 0.0,
        -sinY, 0.0, cosY));
    const cv::Mat rotationZ = cv::Mat(cv::Matx33d(
        cosZ, -sinZ, 0.0,
        sinZ, cosZ, 0.0,
        0.0, 0.0, 1.0));
    return rotationZ * rotationY * rotationX;
}

cv::Vec3d rotationMatrixToEulerAngles(
    const cv::Mat& input,
    const double validationTolerance = kDefaultTolerance,
    const double singularEpsilon = kDefaultTolerance) {
    const cv::Mat rotation = asDoubleRotationMatrix(input);
    if (!isRotationMatrix(rotation, validationTolerance)) {
        throw std::invalid_argument("input is not a proper rotation matrix");
    }
    if (!(singularEpsilon > 0.0)) {
        throw std::invalid_argument("singular epsilon must be positive");
    }

    const double horizontalNorm =
        std::hypot(rotation.at<double>(0, 0), rotation.at<double>(1, 0));
    const bool singular = horizontalNorm < singularEpsilon;

    double xAngle = 0.0;
    double yAngle = 0.0;
    double zAngle = 0.0;
    if (!singular) {
        xAngle = std::atan2(rotation.at<double>(2, 1), rotation.at<double>(2, 2));
        yAngle = std::atan2(-rotation.at<double>(2, 0), horizontalNorm);
        zAngle = std::atan2(rotation.at<double>(1, 0), rotation.at<double>(0, 0));
    } else {
        xAngle = std::atan2(-rotation.at<double>(1, 2), rotation.at<double>(1, 1));
        yAngle = std::atan2(-rotation.at<double>(2, 0), horizontalNorm);
        zAngle = 0.0;
    }
    return cv::Vec3d(xAngle, yAngle, zAngle);
}

double matrixRoundTripError(const cv::Vec3d& angles) {
    const cv::Mat original = eulerAnglesToRotationMatrix(angles);
    const cv::Vec3d recoveredAngles = rotationMatrixToEulerAngles(original);
    const cv::Mat recovered = eulerAnglesToRotationMatrix(recoveredAngles);
    return cv::norm(original - recovered, cv::NORM_INF);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void runValidation() {
    const std::array<cv::Vec3d, 6> fixedCases = {
        cv::Vec3d(0.0, 0.0, 0.0),
        cv::Vec3d(kPi / 2.0, 0.0, 0.0),
        cv::Vec3d(0.0, kPi / 2.0, 0.0),
        cv::Vec3d(0.0, -kPi / 2.0, 0.0),
        cv::Vec3d(0.3, -0.7, 1.2),
        cv::Vec3d(-2.4, kPi / 2.0 - 1e-10, 0.8),
    };

    double maximumError = 0.0;
    for (const cv::Vec3d& angles : fixedCases) {
        maximumError = std::max(maximumError, matrixRoundTripError(angles));
    }

    std::mt19937 generator(20260726U);
    std::uniform_real_distribution<double> angleDistribution(-kPi, kPi);
    constexpr int kRandomCaseCount = 512;
    for (int index = 0; index < kRandomCaseCount; ++index) {
        const cv::Vec3d angles(
            angleDistribution(generator),
            angleDistribution(generator),
            angleDistribution(generator));
        maximumError = std::max(maximumError, matrixRoundTripError(angles));
    }

    const cv::Mat reflection = cv::Mat(cv::Matx33d(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, -1.0));
    require(!isRotationMatrix(reflection), "reflection matrix was incorrectly accepted");
    require(maximumError <= 1e-9, "matrix round-trip error exceeded tolerance");

    const int totalCases = static_cast<int>(fixedCases.size()) + kRandomCaseCount;
    std::cout << "VALIDATION PASSED: " << totalCases
              << " cases, max_matrix_error=" << std::scientific << maximumError << '\n';
}

double parseDouble(const std::string& text) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid finite number: " + text);
    }
    return value;
}

struct Options {
    bool validate = false;
    bool degrees = false;
    bool help = false;
    bool hasAngles = false;
    bool hasMatrix = false;
    cv::Vec3d angles;
    cv::Mat matrix;
};

Options parseArguments(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--degrees") {
            options.degrees = true;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--angles") {
            if (index + 3 >= argc) {
                throw std::invalid_argument("--angles requires X Y Z");
            }
            const double xAngle = parseDouble(argv[++index]);
            const double yAngle = parseDouble(argv[++index]);
            const double zAngle = parseDouble(argv[++index]);
            options.angles = cv::Vec3d(xAngle, yAngle, zAngle);
            options.hasAngles = true;
        } else if (argument == "--matrix") {
            if (index + 9 >= argc) {
                throw std::invalid_argument("--matrix requires nine row-major values");
            }
            options.matrix = cv::Mat(3, 3, CV_64F);
            for (int element = 0; element < 9; ++element) {
                options.matrix.at<double>(element / 3, element % 3) =
                    parseDouble(argv[++index]);
            }
            options.hasMatrix = true;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    const int selectedModes = static_cast<int>(options.validate) +
                              static_cast<int>(options.hasAngles) +
                              static_cast<int>(options.hasMatrix);
    if (selectedModes > 1) {
        throw std::invalid_argument("choose only one of --validate, --angles, or --matrix");
    }
    return options;
}

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program << " [--degrees] [--angles X Y Z | --matrix R00 ... R22 | --validate]\n"
        << "Convention: active right-handed column-vector rotation R = Rz @ Ry @ Rx.\n";
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        if (options.help) {
            printUsage(argv[0]);
            return 0;
        }
        if (options.validate) {
            runValidation();
            return 0;
        }

        cv::Vec3d angles;
        cv::Mat rotation;
        if (options.hasMatrix) {
            rotation = asDoubleRotationMatrix(options.matrix);
            angles = rotationMatrixToEulerAngles(rotation);
        } else {
            angles = options.hasAngles ? options.angles : cv::Vec3d(20.0, -35.0, 70.0);
            if (!options.hasAngles || options.degrees) {
                angles *= kPi / 180.0;
            }
            rotation = eulerAnglesToRotationMatrix(angles);
        }

        const cv::Vec3d displayedAngles =
            options.degrees ? angles * (180.0 / kPi) : angles;
        std::cout << std::setprecision(10);
        std::cout << "Euler angles [x, y, z] ("
                  << (options.degrees ? "degrees" : "radians") << "):\n"
                  << displayedAngles << "\n";
        std::cout << "Rotation matrix Rz * Ry * Rx:\n" << rotation << "\n";

        const cv::Vec3d recoveredAngles = rotationMatrixToEulerAngles(rotation);
        const cv::Mat recovered = eulerAnglesToRotationMatrix(recoveredAngles);
        const double error = cv::norm(rotation - recovered, cv::NORM_INF);
        std::cout << "Round-trip matrix infinity-norm error: "
                  << std::scientific << error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        printUsage(argv[0]);
        return 2;
    }
}
