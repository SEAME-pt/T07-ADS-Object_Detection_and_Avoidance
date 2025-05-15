cmake_minimum_required(VERSION 3.10)

if(NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
    )
endif()

# Nome e versão do projeto
project(MainCarServer VERSION 1.0)

# Configuração do padrão C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/cmake")

# Adicionar o diretório cppservice como subdiretório
# add_subdirectory(cppservice)
add_subdirectory(cppservice/modules/vehicle)
# Especificar diretórios de inclusão para o seu projeto principal
include_directories(includes)
include_directories(includes/Subscribers)
include_directories(${CMAKE_SOURCE_DIR}/cppservice/modules/vehicle/implementation)

# Definir os arquivos fonte do projeto principal
set(MAINCAR_SOURCES
    sources/Car/ACar.cpp
    sources/Car/JetSnailsCar.cpp
    sources/CANBus/CANBus.cpp
    sources/Sensors/ParkSensor.cpp
    sources/Sensors/SpeedSensor.cpp
    sources/Sensors/Odometer.cpp
    sources/main.cpp
)

# Criar o executável para o projeto
add_executable(${PROJECT_NAME} ${MAINCAR_SOURCES})

target_link_libraries(MainCarServer PRIVATE vehicle-core vehicle-implementation)

# Adicionar o diretório de inclusão do ZeroMQ
target_include_directories(${PROJECT_NAME} PRIVATE /usr/include)

# Linkar o ZeroMQ corretamente
target_link_libraries(${PROJECT_NAME} PRIVATE /usr/lib/aarch64-linux-gnu/libzmq.so)

# Procurar a biblioteca de threads (pthread)
find_package(Threads REQUIRED)

# Linkar a biblioteca de threads (pthread) ao seu projeto
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
