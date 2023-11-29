cmake_minimum_required(VERSION 3.10)

project(CudaProject)

# Find CUDA
find_package(CUDA REQUIRED)

# Set source files
set(SOURCES
    kernel.cu
)

# Add executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES})

# Include CUDA headers
include_directories(${CUDA_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
