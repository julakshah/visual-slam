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

find_package(OpenCV REQUIRED)

# Include header files
include_directories(include)

# Create executable from source files
add_executable(orb_demo
    src/main.cpp
    src/orb.cpp
)

# Link libraries
target_link_libraries(orb_demo PRIVATE ${OpenCV_LIBS})
