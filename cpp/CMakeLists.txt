cmake_minimum_required(VERSION 3.18)
project(edgellm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Locate ONNX Runtime. Priority: ORT_HOME env var, then Homebrew (macOS).
if(DEFINED ENV{ORT_HOME})
  set(ORT_ROOT $ENV{ORT_HOME})
elseif(APPLE)
  execute_process(
    COMMAND brew --prefix onnxruntime
    OUTPUT_VARIABLE ORT_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ORT_ROOT}/include ${ORT_ROOT}/include/onnxruntime
  PATH_SUFFIXES onnxruntime)
find_library(ORT_LIBRARY onnxruntime HINTS ${ORT_ROOT}/lib)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime not found. Install it (macOS: 'brew install onnxruntime') "
    "or set ORT_HOME to a directory containing include/ and lib/.")
endif()

message(STATUS "ONNX Runtime headers: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ORT_LIBRARY}")

add_executable(edgellm_infer src/main.cpp)
target_include_directories(edgellm_infer PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(edgellm_infer PRIVATE ${ORT_LIBRARY})
