#include <exception>
#include <iostream>

#include "homography_utils.hpp"

int main(int argc, char** argv) {
    try {
        return homography_example::runComposite(
            argc, argv, "homography-composite.jpg");
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        homography_example::printCompositeUsage(argv[0]);
        return 1;
    }
}
