cmake_minimum_required(VERSION 3.10)
project(orb_project LANGUAGES CXX)

# Use modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    add_compile_options(-O0 -g -fno-omit-frame-pointer -fsanitize=undefined)
    add_link_options(-fno-omit-frame-pointer -fsanitize=undefined)
    add_compile_definitions(PYBIND11_DETAILED_ERROR_MESSAGES PYBIND11_ASSERT)
elseif(MSVC)
    add_compile_options(/W4)
endif()

add_compile_definitions(PYBIND11_DETAILED_ERROR_MESSAGES)
add_compile_definitions(PYBIND11_ASSERT)

# Find the specific OpenCV components we need
find_package(OpenCV REQUIRED COMPONENTS
    core         # For cv::Mat, cv::KeyPoint
    features2d   # For cv::FAST
    imgproc      # For cv::drawKeypoints, cvRound
    highgui      # For cv::imshow, cv::waitKey
    imgcodecs    # For cv::imread
    videoio      # For cv::VideoCapture
)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

pybind11_add_module(orb_project src/main_interface.cpp src/orb.cpp)
target_link_libraries(orb_project PRIVATE ${OpenCV_LIBS})
set_target_properties(orb_project PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Include your local header files ("include/orb.hpp")
include_directories(include)

# Add the OpenCV header file directories to the compiler path
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable from source files
add_executable(orb_demo
    src/main.cpp
    src/orb.cpp
)

# Link libraries
# ${OpenCV_LIBS} will now contain all the components we found
target_link_libraries(orb_demo PRIVATE ${OpenCV_LIBS})