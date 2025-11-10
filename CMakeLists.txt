cmake_minimum_required(VERSION 3.10)
project(orb_project LANGUAGES CXX)

# Use modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# --- START OF FIX ---

# Find the specific OpenCV components we need
find_package(OpenCV REQUIRED COMPONENTS
    core         # For cv::Mat, cv::KeyPoint
    features2d   # For cv::FAST
    imgproc      # For cv::drawKeypoints, cvRound
    highgui      # For cv::imshow, cv::waitKey
    imgcodecs    # For cv::imread
    videoio      # For cv::VideoCapture
)

# Include your local header files ("include/orb.hpp")
include_directories(include)

# !! THIS IS THE MISSING LINE !!
# Add the OpenCV header file directories to the compiler path
include_directories(${OpenCV_INCLUDE_DIRS})

# --- END OF FIX ---


# Create executable from source files
add_executable(orb_demo
    src/main.cpp
    src/orb.cpp
)

# Link libraries
# ${OpenCV_LIBS} will now contain all the components we found
target_link_libraries(orb_demo PRIVATE ${OpenCV_LIBS})