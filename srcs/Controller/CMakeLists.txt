cmake_minimum_required(VERSION 3.10)
project(Controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Otimização e warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Includes do projeto e sysroot
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen

    /opt/jetson-sysroot/usr/include
    /opt/jetson-sysroot/usr/include/aarch64-linux-gnu
    /opt/jetson-sysroot/usr/local/cuda/include
    /opt/jetson-sysroot/usr/local/include/opencv4
    /opt/jetson-sysroot/usr/local/include/opencv4/opencv2
    /opt/jetson-sysroot/usr/src/tensorrt/include
    /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/tegra

    /opt/jetson-sysroot/usr/include/SDL2
)

# pkg-config para libgpiod
find_package(PkgConfig REQUIRED)
pkg_check_modules(GPIOD REQUIRED libgpiod)

# Bibliotecas externas no sysroot
find_library(I2C_LIB             i2c              PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(NVINFER            nvinfer          PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(NVONNXPARSER       nvonnxparser     PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(CUDART             cudart           PATHS /opt/jetson-sysroot/usr/local/cuda/lib64)
find_library(ZMQ                zmq              PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(SDL2_LIB           SDL2             PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(NVMEDIA            nvmedia          PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/tegra)

# OpenCV módulos principais
find_library(OpenCV_CORE        opencv_core      PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_IMGPROC     opencv_imgproc   PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_HIGHGUI     opencv_highgui   PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_VIDEOIO     opencv_videoio   PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_VIDEO       opencv_video     PATHS /opt/jetson-sysroot/usr/local/lib)       # Para KalmanFilter
find_library(OpenCV_CUDAWARPING opencv_cudawarping PATHS /opt/jetson-sysroot/usr/local/lib)     # Para cv::cuda::resize

# Fontes do projeto
set(SOURCES
    sources/Controller.cpp
    sources/JetCar.cpp
    sources/devices.cpp
    sources/LaneDetector.cpp
    sources/main.cpp
)

# Executável
add_executable(ControllerExec ${SOURCES})

# Linkagem
target_link_libraries(ControllerExec
    ${GPIOD_LIBRARIES}
    ${I2C_LIB}
    ${NVINFER}
    ${NVONNXPARSER}
    ${CUDART}
    ${ZMQ}
    ${SDL2_LIB}
    ${NVMEDIA}

    ${OpenCV_CORE}
    ${OpenCV_IMGPROC}
    ${OpenCV_HIGHGUI}
    ${OpenCV_VIDEOIO}
    ${OpenCV_VIDEO}
    ${OpenCV_CUDAWARPING}
)
