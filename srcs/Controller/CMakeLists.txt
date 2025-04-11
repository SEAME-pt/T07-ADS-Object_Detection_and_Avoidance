cmake_minimum_required(VERSION 3.10)

# Nome do projeto
project(Controller)

# Definição do padrão C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adiciona flags de otimização para a Jetson Nano
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Diretórios de inclusão
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen  # Inclui o diretório eigen na mesma pasta do CMake
    /usr/include/aarch64-linux-gnu
    /usr/local/cuda/include
    /usr/include/opencv4
    /usr/src/tensorrt/include
)

# Procurar pacotes externos
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(SDL2 REQUIRED)

# Verifica se gpiod e i2c-dev estão instalados
pkg_check_modules(GPIOD REQUIRED libgpiod)
find_library(I2C_LIB i2c)

# Adiciona TensorRT
find_library(NVINFER nvinfer)
find_library(NVONNXPARSER nvonnxparser)
find_library(CUDART cudart HINTS /usr/local/cuda/lib64)

# Adiciona as fontes do projeto
set(SOURCES
    sources/Controller.cpp
    sources/JetCar.cpp
    sources/devices.cpp
    sources/LaneDetector.cpp
    sources/main.cpp
)

# Criar o executável
add_executable(ControllerExec ${SOURCES})

# Incluir diretórios das bibliotecas externas
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${GPIOD_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
)

# Vincular as bibliotecas necessárias
target_link_libraries(ControllerExec
    ${OpenCV_LIBS}
    ${GPIOD_LIBRARIES}
    ${I2C_LIB}
    ${NVINFER}
    ${NVONNXPARSER}
    ${CUDART}
    ${SDL2_LIBRARIES}
    /usr/lib/aarch64-linux-gnu/libzmq.so  # ZeroMQ
)
