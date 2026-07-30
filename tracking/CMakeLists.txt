cmake_minimum_required(VERSION 3.16)
project(object_tracking_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED
    COMPONENTS core dnn highgui imgcodecs imgproc video videoio
    OPTIONAL_COMPONENTS tracking
)
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
    message(FATAL_ERROR
        "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}"
    )
endif()

set(
    TRACKING_MODELS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/models"
    CACHE PATH
    "Directory containing the verified ONNX tracker models"
)

add_executable(object_tracker tracker.cpp)
target_link_libraries(object_tracker PRIVATE ${OpenCV_LIBS})
target_compile_options(object_tracker PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic;-Werror>
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME tracking_smoke
        COMMAND object_tracker
            --input=${CMAKE_CURRENT_SOURCE_DIR}/videos/chaplin.mp4
            --tracker=MIL
            --bbox=287,23,86,320
            --max-frames=5
            --snapshot=${CMAKE_CURRENT_BINARY_DIR}/tracking-smoke.png
    )

    if(
        EXISTS "${TRACKING_MODELS_DIR}/dasiamrpn_model.onnx"
        AND EXISTS "${TRACKING_MODELS_DIR}/dasiamrpn_kernel_cls1.onnx"
        AND EXISTS "${TRACKING_MODELS_DIR}/dasiamrpn_kernel_r1.onnx"
    )
        add_test(
            NAME tracking_dasiamrpn_smoke
            COMMAND object_tracker
                --input=${CMAKE_CURRENT_SOURCE_DIR}/videos/chaplin.mp4
                --tracker=DASIAMRPN
                --models-dir=${TRACKING_MODELS_DIR}
                --bbox=287,23,86,320
                --max-frames=5
                --snapshot=${CMAKE_CURRENT_BINARY_DIR}/tracking-dasiamrpn.png
        )
        set_tests_properties(tracking_dasiamrpn_smoke PROPERTIES TIMEOUT 300)
    endif()

    if(
        EXISTS "${TRACKING_MODELS_DIR}/nanotrack_backbone_sim.onnx"
        AND EXISTS "${TRACKING_MODELS_DIR}/nanotrack_head_sim.onnx"
    )
        add_test(
            NAME tracking_nano_smoke
            COMMAND object_tracker
                --input=${CMAKE_CURRENT_SOURCE_DIR}/videos/chaplin.mp4
                --tracker=NANO
                --models-dir=${TRACKING_MODELS_DIR}
                --bbox=287,23,86,320
                --max-frames=5
                --snapshot=${CMAKE_CURRENT_BINARY_DIR}/tracking-nano.png
        )
        set_tests_properties(tracking_nano_smoke PROPERTIES TIMEOUT 300)
    endif()

    if(EXISTS "${TRACKING_MODELS_DIR}/object_tracking_vittrack_2023sep.onnx")
        add_test(
            NAME tracking_vit_smoke
            COMMAND object_tracker
                --input=${CMAKE_CURRENT_SOURCE_DIR}/videos/chaplin.mp4
                --tracker=VIT
                --models-dir=${TRACKING_MODELS_DIR}
                --bbox=250,15,180,340
                --max-frames=5
                --snapshot=${CMAKE_CURRENT_BINARY_DIR}/tracking-vit.png
        )
        set_tests_properties(tracking_vit_smoke PROPERTIES TIMEOUT 300)
    endif()
endif()
