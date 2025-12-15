cmake_minimum_required(VERSION 3.16)
project(Five_Threads LANGUAGES C CXX)

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
add_definitions(${GST_CFLAGS_OTHER})

add_executable(Five_Threads
        src/main.cpp
        src/rtsp_worker.cpp
        src/frame_store.cpp
        src/detector.cpp
        src/tracker_manager.cpp
        src/overlay.cpp
        src/offset_calculator.cpp
)

target_link_libraries(Five_Threads
        ${GST_LIBRARIES}
        ${OpenCV_LIBS}
        pthread
)
