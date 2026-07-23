#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#else
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#endif
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifndef IMAGE_ALIGNMENT_SOURCE_DIR
#define IMAGE_ALIGNMENT_SOURCE_DIR "."
#endif

namespace {

constexpr int kMaxFeatures = 500;
constexpr float kGoodMatchPercent = 0.15F;
constexpr std::size_t kMinHomographyMatches = 4;

struct AlignmentStats {
  std::size_t inputKeypoints;
  std::size_t referenceKeypoints;
  std::size_t totalMatches;
  std::size_t retainedMatches;
  int inlierMatches;
};

void writeImage(const std::string& filename, const cv::Mat& image) {
  const std::filesystem::path outputPath(filename);
  if (outputPath.has_parent_path()) {
    std::filesystem::create_directories(outputPath.parent_path());
  }
  if (!cv::imwrite(filename, image)) {
    throw std::runtime_error("Could not write image: " + filename);
  }
}

AlignmentStats alignImagesImpl(const cv::Mat& image, const cv::Mat& reference,
                               cv::Mat& aligned, cv::Mat& homography,
                               const std::string& matchesFilename) {
  if (image.empty()) {
    throw std::invalid_argument("The image to align is empty");
  }
  if (reference.empty()) {
    throw std::invalid_argument("The reference image is empty");
  }

  cv::Mat imageGray;
  cv::Mat referenceGray;
  cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(reference, referenceGray, cv::COLOR_BGR2GRAY);

  std::vector<cv::KeyPoint> keypoints1;
  std::vector<cv::KeyPoint> keypoints2;
  cv::Mat descriptors1;
  cv::Mat descriptors2;

  const cv::Ptr<cv::Feature2D> orb = cv::ORB::create(kMaxFeatures);
  orb->detectAndCompute(imageGray, cv::noArray(), keypoints1, descriptors1);
  orb->detectAndCompute(referenceGray, cv::noArray(), keypoints2, descriptors2);

  if (descriptors1.empty() || descriptors2.empty()) {
    throw std::runtime_error("ORB could not find features in both images");
  }

  std::vector<cv::DMatch> matches;
  const cv::Ptr<cv::BFMatcher> matcher =
      cv::BFMatcher::create(cv::NORM_HAMMING, false);
  matcher->match(descriptors1, descriptors2, matches);

  if (matches.size() < kMinHomographyMatches) {
    throw std::runtime_error("At least four feature matches are required");
  }

  const auto matchSortKey = [&keypoints1, &keypoints2](const cv::DMatch& match) {
    const cv::Point2f point1 = keypoints1[match.queryIdx].pt;
    const cv::Point2f point2 = keypoints2[match.trainIdx].pt;
    return std::make_tuple(match.distance, point1.y, point1.x, point2.y,
                           point2.x, match.queryIdx, match.trainIdx,
                           match.imgIdx);
  };
  std::sort(matches.begin(), matches.end(),
            [&matchSortKey](const cv::DMatch& left, const cv::DMatch& right) {
              return matchSortKey(left) < matchSortKey(right);
            });

  const std::size_t totalMatchCount = matches.size();
  const auto nominalMatchCount =
      static_cast<std::size_t>(matches.size() * kGoodMatchPercent);
  const auto minimumMatchCount =
      std::max(kMinHomographyMatches, nominalMatchCount);
  const float cutoffDistance = matches[minimumMatchCount - 1].distance;
  const auto retainedEnd =
      std::find_if(matches.begin(), matches.end(),
                   [cutoffDistance](const cv::DMatch& match) {
                     return match.distance > cutoffDistance;
                   });
  matches.erase(retainedEnd, matches.end());

  if (!matchesFilename.empty()) {
    cv::Mat matchesImage;
    cv::drawMatches(image, keypoints1, reference, keypoints2, matches,
                    matchesImage, cv::Scalar(0, 255, 0),
                    cv::Scalar(255, 0, 0), std::vector<char>(),
                    cv::DrawMatchesFlags::DEFAULT);
    writeImage(matchesFilename, matchesImage);
  }

  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;
  points1.reserve(matches.size());
  points2.reserve(matches.size());
  for (const auto& match : matches) {
    points1.push_back(keypoints1[match.queryIdx].pt);
    points2.push_back(keypoints2[match.trainIdx].pt);
  }

  cv::Mat inlierMask;
  homography =
      cv::findHomography(points1, points2, cv::RANSAC, 3.0, inlierMask);
  if (homography.empty() || !cv::checkRange(homography)) {
    throw std::runtime_error("Could not estimate a valid homography");
  }
  if (inlierMask.empty()) {
    throw std::runtime_error("Could not determine homography inliers");
  }

  cv::warpPerspective(image, aligned, homography, reference.size());
  return AlignmentStats{
      keypoints1.size(),
      keypoints2.size(),
      totalMatchCount,
      matches.size(),
      cv::countNonZero(inlierMask),
  };
}

cv::Mat readImage(const std::string& filename, const std::string& description) {
  cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not read " + description + ": " + filename);
  }
  return image;
}

double meanAbsoluteError(const cv::Mat& first, const cv::Mat& second) {
  if (first.size() != second.size() || first.type() != second.type()) {
    throw std::invalid_argument("Images must have the same size and type");
  }

  cv::Mat difference;
  cv::absdiff(first, second, difference);
  const cv::Scalar channelMeans = cv::mean(difference);
  double total = 0.0;
  for (int channel = 0; channel < difference.channels(); ++channel) {
    total += channelMeans[channel];
  }
  return total / difference.channels();
}

void validateAlignment(const cv::Mat& image, const cv::Mat& reference,
                       const cv::Mat& aligned) {
  if (aligned.size() != reference.size() ||
      aligned.type() != reference.type()) {
    throw std::runtime_error(
        "Aligned image dimensions or type do not match the reference");
  }

  cv::Mat resizedInput;
  cv::resize(image, resizedInput, reference.size());
  const double before = meanAbsoluteError(resizedInput, reference);
  const double after = meanAbsoluteError(aligned, reference);

  std::cout << "Alignment MAE before: " << before << '\n';
  std::cout << "Alignment MAE after: " << after << '\n';
  if (!std::isfinite(before) || !std::isfinite(after) || before <= 0.0 ||
      after >= before * 0.4) {
    throw std::runtime_error(
        "Alignment did not improve mean absolute error by the required amount");
  }
}

void validateWrittenOutputs(const std::string& alignedFilename,
                            const std::string& matchesFilename,
                            const cv::Size& alignedSize,
                            const cv::Size& matchesSize) {
  const cv::Mat aligned =
      readImage(alignedFilename, "written aligned image");
  const cv::Mat matches =
      readImage(matchesFilename, "written feature matches image");
  if (aligned.size() != alignedSize) {
    throw std::runtime_error(
        "Written aligned image dimensions do not match the reference");
  }
  if (matches.size() != matchesSize) {
    throw std::runtime_error(
        "Written feature matches dimensions are not as expected");
  }
}

struct Arguments {
  std::string inputFilename =
      (std::filesystem::path(IMAGE_ALIGNMENT_SOURCE_DIR) / "scanned-form.jpg")
          .string();
  std::string referenceFilename =
      (std::filesystem::path(IMAGE_ALIGNMENT_SOURCE_DIR) / "form.jpg").string();
  std::filesystem::path outputDirectory = ".";
  bool validate = false;
  bool showHelp = false;
};

Arguments parseArguments(int argc, char** argv) {
  Arguments arguments;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string option(argv[argument]);
    const auto requireValue = [&]() -> std::string {
      if (argument + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + option);
      }
      return argv[++argument];
    };

    if (option == "--input") {
      arguments.inputFilename = requireValue();
    } else if (option == "--reference") {
      arguments.referenceFilename = requireValue();
    } else if (option == "--output-dir") {
      arguments.outputDirectory = requireValue();
    } else if (option == "--validate") {
      arguments.validate = true;
    } else if (option == "--no-display") {
      // This example has no GUI; accept the shared headless-example option.
    } else if (option == "--help" || option == "-h") {
      arguments.showHelp = true;
    } else {
      throw std::invalid_argument("Unknown argument: " + option);
    }
  }
  return arguments;
}

void printUsage(const char* programName) {
  std::cout
      << "Usage: " << programName
      << " [--input PATH] [--reference PATH] [--output-dir PATH]\n"
         "       [--no-display] [--validate]\n";
}

}  // namespace

// Preserve the original tutorial function signature for readers who copied it
// into another C++ program. The command-line example below uses the extended
// implementation so callers can choose the matches output path.
void alignImages(cv::Mat& image, cv::Mat& reference, cv::Mat& aligned,
                 cv::Mat& homography) {
  static_cast<void>(
      alignImagesImpl(image, reference, aligned, homography, "matches.jpg"));
}

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parseArguments(argc, argv);
    if (arguments.showHelp) {
      printUsage(argv[0]);
      return 0;
    }

    const std::string outputFilename =
        (arguments.outputDirectory / "aligned.jpg").string();
    const std::string matchesFilename =
        (arguments.outputDirectory / "matches.jpg").string();
    std::cout << "OpenCV version: " << CV_VERSION << '\n';
    std::cout << "Reading reference image: " << arguments.referenceFilename
              << '\n';
    const cv::Mat reference =
        readImage(arguments.referenceFilename, "reference image");
    std::cout << "Reading image to align: " << arguments.inputFilename << '\n';
    const cv::Mat image =
        readImage(arguments.inputFilename, "input image");

    cv::Mat aligned;
    cv::Mat homography;
    std::cout << "Aligning images ...\n";
    const AlignmentStats stats = alignImagesImpl(
        image, reference, aligned, homography, matchesFilename);
    std::cout << "Detected keypoints: input=" << stats.inputKeypoints
              << ", reference=" << stats.referenceKeypoints << '\n';
    std::cout << "Feature matches: total=" << stats.totalMatches
              << ", retained=" << stats.retainedMatches
              << ", inliers=" << stats.inlierMatches << '\n';
    if (arguments.validate) {
      validateAlignment(image, reference, aligned);
    }
    writeImage(outputFilename, aligned);
    if (arguments.validate) {
      const cv::Size matchesSize(
          image.cols + reference.cols,
          std::max(image.rows, reference.rows));
      validateWrittenOutputs(outputFilename, matchesFilename, reference.size(),
                             matchesSize);
      std::cout << "Alignment validation passed\n";
    }

    std::cout << "Saved aligned image: " << outputFilename << '\n';
    std::cout << "Saved feature matches: " << matchesFilename << '\n';
    std::cout << "Estimated homography:\n" << homography << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
