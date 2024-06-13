cmake_minimum_required(VERSION 2.6)
project(svm_cpp)


# The library prefix
SET(LIB_PREFIX _svm_test)

set(CMAKE_CXX_FLAGS "-g -Wall")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${svm_test_SOURCE_DIR}/bin)

ADD_LIBRARY(svm_lib
        svm.h svm.cpp
)
ADD_EXECUTABLE(svm_classification SVM_Classification.cpp)
ADD_EXECUTABLE(svm_regression SVM_Regression.cpp)

SET_TARGET_PROPERTIES(svm_lib PROPERTIES OUTPUT_NAME ${LIB_PREFIX}_svm_lib)
SET_TARGET_PROPERTIES(svm_classification PROPERTIES OUTPUT_NAME svm_classification)
SET_TARGET_PROPERTIES(svm_regression PROPERTIES OUTPUT_NAME svm_regression)

# link the library to the executable
TARGET_LINK_LIBRARIES(svm_classification svm_lib)
TARGET_LINK_LIBRARIES(svm_regression svm_lib)
