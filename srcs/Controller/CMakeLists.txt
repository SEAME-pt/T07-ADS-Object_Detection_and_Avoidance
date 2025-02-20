cmake_minimum_required(VERSION 3.10)

# Defina o nome do seu projeto
project(Controller)

# Defina o padrão C++ para C++17
set(CMAKE_CXX_STANDARD 17)

# Definindo diretórios de inclusão para as bibliotecas externas
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)

# Encontrar pacotes externos
find_package(SDL2 REQUIRED)
find_package(PkgConfig REQUIRED)

# Verifique se gpiod e i2c-dev estão instalados
pkg_check_modules(GPIOD REQUIRED libgpiod)
find_library(I2C_LIB i2c)

# Adicionar as fontes do projeto
set(SOURCES
    sources/Controller.cpp
    sources/JetCar.cpp
    sources/devices.cpp
    sources/main.cpp
)

# Criar o executável
add_executable(ControllerExec ${SOURCES})

# Incluir diretórios das bibliotecas externas
include_directories(${SDL2_INCLUDE_DIRS} ${GPIOD_INCLUDE_DIRS})

# Vincular as bibliotecas necessárias
target_link_libraries(ControllerExec
    ${SDL2_LIBRARIES}
    ${GPIOD_LIBRARIES}
    ${I2C_LIB}
    /usr/lib/aarch64-linux-gnu/libzmq.so  # Bibliotecas do ZeroMQ, se necessário
)
