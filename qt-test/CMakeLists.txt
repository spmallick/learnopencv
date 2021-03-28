cmake_minimum_required(VERSION 3.5.1)

project(sample)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(sample main.cpp camviewer.cpp mainwindow.cpp mainwindow.ui)
target_include_directories(sample PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sample PRIVATE  Qt5::Widgets ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
