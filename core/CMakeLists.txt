cmake_minimum_required(VERSION 3.20)
project(bai_core LANGUAGES CXX C VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include project headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Discover pybind11 from the active Python environment instead of requiring a system install
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_CMAKE_DIR)
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake directory")
endif()
find_package(pybind11 CONFIG QUIET)

# Core library sources
set(CORE_SOURCES
    src/engine.cpp
    src/c_api.cpp
    src/tokenizer.cpp
    src/json_builder.cpp
    src/inference_engine.cpp
)

# Build the static core library even if the Python bindings are unavailable.
add_library(bai_static STATIC ${CORE_SOURCES})

target_include_directories(bai_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation prefix")
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
)
if(NOT ONNXRUNTIME_LIB)
    find_file(ONNXRUNTIME_LIB
        NAMES libonnxruntime.so.1.29.0
        HINTS "${ONNXRUNTIME_ROOT}" "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
    )
endif()
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found. Set ONNXRUNTIME_ROOT or provide "
        "ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIB."
    )
endif()

target_include_directories(bai_static PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(bai_static PRIVATE ${ONNXRUNTIME_LIB})

# Python module is optional because the local environment may not expose a CMake config.
if(pybind11_FOUND)
    pybind11_add_module(bai_core bindings/bai_pybind.cpp ${CORE_SOURCES})
    target_include_directories(bai_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(bai_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(bai_core PRIVATE ${ONNXRUNTIME_LIB})
else()
    message(WARNING "pybind11 not found via CMake; skipping bai_core Python extension target.")
endif()