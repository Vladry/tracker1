cmake_minimum_required(VERSION 3.16)
project(Five_Threads LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

find_package(OpenCV REQUIRED)

add_executable(Five_Threads)

configure_file(
        ${CMAKE_SOURCE_DIR}/config.toml
        ${CMAKE_BINARY_DIR}/config.toml
        COPYONLY
)

target_sources(Five_Threads PRIVATE
        src/detector.cpp
        src/display_loop.cpp
        src/frame_store.cpp
        src/main.cpp
        src/offset_calculator.cpp
        src/overlay.cpp
        src/rtsp_worker.cpp
        src/rtsp_watch_dog.cpp
        src/static_box_manager.cpp
        src/tracker_manager.cpp
        src/merge_bbox.cpp
)

target_include_directories(Five_Threads PRIVATE
        ${GST_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include
)

target_link_libraries(Five_Threads PRIVATE
        ${GST_LIBRARIES}
        ${OpenCV_LIBS}
        pthread
)

add_definitions(${GST_CFLAGS_OTHER})
link_directories(${GST_LIBRARY_DIRS})
