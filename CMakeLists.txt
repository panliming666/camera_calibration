cmake_minimum_required(VERSION 3.16)
project(camera_calibration_service)

# Check if the compiler is a C++ compiler
if(NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found. Please install a C++ compiler.")
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# If the CXX_STANDARD is set to auto, fail if a compiler doesn't support at least C++17
if(CMAKE_CXX_STANDARD STREQUAL "auto")
  message(FATAL_ERROR "CMAKE_CXX_STANDARD is set to 'auto', but a compiler that supports at least C++17 is required.")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Set the message files to be generated
set(msg_files
  "msg/ActionCalibration.msg"
  "msg/CalibrationInformation.msg"
)

# Generate the messages
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)

# This is an ament package
ament_package()
