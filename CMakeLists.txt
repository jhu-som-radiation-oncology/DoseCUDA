cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)

add_subdirectory(DoseCUDA/dose_kernels)

install(TARGETS dose_kernels DESTINATION ${SKBUILD_PROJECT_NAME})
