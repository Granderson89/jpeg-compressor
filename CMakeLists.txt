cmake_minimum_required(VERSION 3.10)
project(jpeg_compression)

set(CMAKE_CXX_STANDARD 11)

# Get packages
find_package(Threads)
link_libraries(${CMAKE_THREAD_LIBS_INIT})

find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_libraries(${OpenCL_LIBRARIES})

# Get source files
file(GLOB_RECURSE SOURCE_FILES *.cpp *.h)

# Create executable
add_executable(jpeg_compression ${SOURCE_FILES})

# Copy images from source to build folder
add_custom_command(TARGET jpeg_compression POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/images" $<TARGET_FILE_DIR:jpeg_compression>/images)

# Copy the kernel files to the build output
configure_file(rgb_to_ycc.cl rgb_to_ycc.cl COPYONLY)