cmake_minimum_required(VERSION 3.16)
project(mlperf_tiny C CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/k230-deploy.cmake)
k230_sdk_paths()

# MLPerf Tiny submodule paths
set(MLPERF_TINY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../mlperf_tiny)
set(MLPERF_API_DIR ${MLPERF_TINY_DIR}/benchmark/api)

# Supported benchmarks
set(_BENCHMARKS ic01 vww01 kws01 ad01)

# Python / venv
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(_VENV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../.venv)
set(_VENV_PYTHON ${_VENV_DIR}/bin/python)
set(_REQUIREMENTS ${CMAKE_CURRENT_SOURCE_DIR}/../../requirements.txt)
set(_KMODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/kmodels)

# kmodel conversion target — all benchmarks
add_custom_target(kmodel
    COMMAND ${Python3_EXECUTABLE} -m venv ${_VENV_DIR}
    COMMAND ${_VENV_PYTHON} -m pip install -r ${_REQUIREMENTS} --quiet
    COMMAND ${_VENV_PYTHON}
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/convert_kmodel.py
            --benchmark ${_BENCHMARKS}
            -o ${_KMODEL_DIR}
    USES_TERMINAL
    COMMENT "Converting TFLite -> kmodel (all benchmarks)"
)

add_executable(mlperf_tiny
    src/main.cc
    src/submitter_implemented.cc
    src/internally_implemented.cpp
)

target_compile_features(mlperf_tiny PRIVATE cxx_std_20)

# Empty TH_MODEL_VERSION to suppress #error in submodule header.
# The actual value is set at runtime via model_version_ global.
target_compile_definitions(mlperf_tiny PRIVATE TH_MODEL_VERSION="")

target_include_directories(mlperf_tiny PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MLPERF_TINY_DIR}/benchmark
    ${_MPP_ROOT}/include
    ${_MPP_ROOT}/include/comm
    ${_MPP_ROOT}/include/ioctl
    ${_MPP_ROOT}/userapps/api
    ${_NNCASE_ROOT}
    ${_NNCASE_ROOT}/nncase/include
    ${_NNCASE_ROOT}/nncase/include/nncase
    ${_NNCASE_ROOT}/nncase/include/nncase/runtime
    ${_NNCASE_ROOT}/rvvlib/include
)

# Warnings for our own sources
target_compile_options(mlperf_tiny PRIVATE
    -Wno-error -Wno-register -Wno-multichar -Wno-pessimizing-move
    -Wno-deprecated-declarations -Wno-unused-result -Wno-unused-variable
    -Wno-format -Wno-return-type -Wno-sign-compare -Wno-unused-label
    -Wno-deprecated-enum-enum-conversion
)

# Suppress warnings for our copy of internally_implemented.cpp (external code)
set_source_files_properties(
    src/internally_implemented.cpp
    PROPERTIES COMPILE_OPTIONS "-w"
)

# RT-Smart SDK includes (set by toolchain file)
if(RTT_INCLUDE)
    target_include_directories(mlperf_tiny PRIVATE ${RTT_INCLUDE})
endif()

if(NOT MPP_LIB_DIR)
    message(FATAL_ERROR "MPP_LIB_DIR not set. Use the bigcore toolchain file.")
endif()

# MPP + nncase libraries (grouped together to resolve circular dependencies)
file(GLOB _MPP_LIBS "${MPP_LIB_DIR}/lib*.a")
list(TRANSFORM _MPP_LIBS REPLACE ".*/lib([^/]*)\\.a$" "-l\\1")
target_link_libraries(mlperf_tiny PRIVATE
    -L${MPP_LIB_DIR}
    -L${_NNCASE_ROOT}/rvvlib
    -L${_NNCASE_ROOT}/nncase/lib
    -Wl,--start-group
        ${_MPP_LIBS}
        -lrvv
        -lNncase.Runtime.Native
        -lnncase.rt_modules.k230
        -lfunctional_k230
    -Wl,--end-group
)

# Deploy target — binary + all kmodels
set(_DEPLOY_DIR /sharefs/mlperf_tiny)
set(_DEPLOY_FILES $<TARGET_FILE:mlperf_tiny>:mlperf_tiny)
foreach(_bench ${_BENCHMARKS})
    list(APPEND _DEPLOY_FILES ${_KMODEL_DIR}/${_bench}.kmodel:${_bench}.kmodel)
endforeach()

k230_add_deploy_target(
    DEPLOY_DIR ${_DEPLOY_DIR}
    DEPENDS mlperf_tiny kmodel
    FILES ${_DEPLOY_FILES}
)

# ---------------------------------------------------------------------------
# ctest — benchmark and golden tests for each benchmark
# ---------------------------------------------------------------------------
enable_testing()
foreach(_bench ${_BENCHMARKS})
    add_test(NAME ${_bench}
        COMMAND ${_VENV_PYTHON}
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_benchmark.py
            --benchmark ${_bench}
            --port ${K230_SERIAL}
            --baud ${K230_BAUD}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/results/${_bench}
    )
    add_test(NAME ${_bench}_golden
        COMMAND ${_VENV_PYTHON}
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/golden_test.py
            --benchmark ${_bench}
            --port ${K230_SERIAL}
            --baud ${K230_BAUD}
    )
endforeach()
