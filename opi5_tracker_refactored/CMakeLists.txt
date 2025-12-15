cmake_minimum_required(VERSION 3.16)
project(opi5_tracker_refactored CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)
find_package(OpenCV REQUIRED)

include_directories(
    ${GST_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(${GST_LIBRARY_DIRS})

add_executable(opi5_tracker
    src/main.cpp
    src/core/frame_store.cpp
    src/rtsp/rtsp_worker.cpp
    src/detect/motion_detector.cpp
    src/blob/blob_merger.cpp
    src/tracker/tracker_manager.cpp
    src/overlay/overlay_renderer.cpp
    src/util/rect_utils.cpp
    src/util/offset_calculator.cpp
)

target_compile_options(opi5_tracker PRIVATE ${GST_CFLAGS_OTHER})
target_link_libraries(opi5_tracker
    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
)
