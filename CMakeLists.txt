cmake_minimum_required(VERSION 3.17)
project(gpu_library LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

include_directories(/usr/local/cuda/include
                    /usr/local/cuda/samples/common/inc
        keg_cuda_lib
        gpu_test_app
        ../../Gene/KGL_Gene/kel_utility                 # Basic Infrastructure library
        ../../Gene/KGL_Gene/kgl_genomics                  # Basic Genetic library
        ../../Gene/KGL_Gene/kgl_mixture                  # Deconvolution and mixture library
        ../../Gene/KGL_Gene/kgl_analytic                  # Genetic analysis library
        ../../Gene/KGL_Gene/kgl_app)

link_directories(../../Gene/KGL_Gene/lib)

set(BOOST_INCLUDEDIR /usr/local/include)
set(BOOST_LIBRARYDIR /usr/local/lib/boost)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME ON)
include_directories(${Boost_INCLUDE_DIRS})
find_package( Boost COMPONENTS system filesystem timer chrono program_options iostreams REQUIRED )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../..)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)


# Debug g++
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC -fconcepts -ggdb -DBOOST_MATH_DISABLE_FLOAT128")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-ggdb -DBOOST_MATH_DISABLE_FLOAT128")

# Release g++
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -fconcepts -O3 -s -DNDEBUG -DBOOST_MATH_DISABLE_FLOAT128")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE " -O3 -s -DNDEBUG -DBOOST_MATH_DISABLE_FLOAT128")

# Profile g++
set(CMAKE_CXX_FLAGS_PROFILE "-fPIC -fconcepts -pg -g -ggdb -O3 -DBOOST_MATH_DISABLE_FLOAT128 -Xlinker -Map=output.map ")
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "-pg -g -ggdb -O3 -DBOOST_MATH_DISABLE_FLOAT128 -Xlinker -Map=output.map ")

# All possible warnings
#set(CMAKE_ERROR_WARNING_FLAGS "-std=c++17 -Wall -pthread -Wextra -Werror -Wpedantic -pedantic-errors -Wstrict-aliasing -Wno-variadic-macros")
# Relax slightly to temporarily allow unused parameters.
set(CMAKE_ERROR_WARNING_FLAGS "-std=c++17 -Wall -pthread -Wextra -Wpedantic -Wstrict-aliasing -Wno-variadic-macros")

# Actually set the g++ flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${CMAKE_ERROR_WARNING_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}  ${CMAKE_ERROR_WARNING_FLAGS} ${CMAKE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_ERROR_WARNING_FLAGS} ${CMAKE_LINKER_FLAGS_RELEASE}")


set(CUDA_LIBRARY_SOURCE_FILES
        keg_cuda_lib/keg_gpu_device_info.cpp
        keg_cuda_lib/keg_gpu_device_info.h
        keg_cuda_lib/keg_gpu_device.cpp
        keg_cuda_lib/keg_gpu_device.h
        keg_cuda_lib/keg_cuda_device.cpp
        keg_cuda_lib/keg_cuda_device.h
        keg_cuda_lib/keg_cuda_cublas.cpp
        keg_cuda_lib/keg_cuda_cublas.h
        keg_cuda_lib/keg_cuda_cublas_matrix.h)

set(CUDA_TEST_APP_FILES
        gpu_test_app/cuda_kernel.cu
        gpu_test_app/ket_gpu_test_run.cpp
        gpu_test_app/ket_gpu_test_main.cpp
        gpu_test_app/ket_gpu_test_app.h
        gpu_test_app/ket_gpu_test_app.cpp
        gpu_test_app/ket_gpu_test_matrix.cpp
        gpu_test_app/ket_gpu_test_matrix.h)


find_package(CUDAToolkit)

add_library(keg_cuda STATIC ${CUDA_LIBRARY_SOURCE_FILES})

target_link_libraries(keg_cuda kel_utility ${Boost_LIBRARIES} ${CUDA_LIBRARIES} cuda cublas cudart nvidia-ml)

add_executable(gpu_test ${CUDA_TEST_APP_FILES} ${CUDA_LIBRARY_SOURCE_FILES})

target_link_libraries(gpu_test keg_cuda)

 set_target_properties(
        gpu_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        PRIVATE "--keep")

target_compile_options(gpu_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_35>)

# Cmake 3.18 property.
#set_property(TARGET gpu_library PROPERTY CUDA_ARCHITECTURES 35)
