cmake_minimum_required(VERSION 3.16)

project(Digital_Cluster VERSION 0.1 LANGUAGES CXX)

# Habilita cobertura de código somente para GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fprofile-arcs -ftest-coverage)
    add_link_options(-fprofile-arcs -lgcov)
endif()

find_package(Qt6 REQUIRED COMPONENTS Core Qml Test)


# Baixar Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.3.2.tar.gz
)
FetchContent_MakeAvailable(Catch2)


# Executável de testes para a classe System
add_executable(tests_system
    test_system.cpp
    ../System.cpp
)

# Ativar AUTOMOC para projetos que usam Qt
set_target_properties(tests_system PROPERTIES AUTOMOC ON)

# Adicionar diretórios de inclusão (depois de criar o executável)
target_include_directories(tests_system PRIVATE ${Catch2_SOURCE_DIR}/src)

# Executável de testes para a classe ZMQReader
add_executable(tests_zmqreader
    test_zmqreader.cpp
    ../zmqreader.cpp
    ../System.cpp
)

# Ativar AUTOMOC para projetos que usam Qt
set_target_properties(tests_zmqreader PROPERTIES AUTOMOC ON)

# Adicionar diretórios de inclusão (depois de criar o executável)
target_include_directories(tests_zmqreader PRIVATE ${Catch2_SOURCE_DIR}/src)

# Executável de testes para a classe System
add_executable(tests_qml
    test_qml_integration.cpp
    ../System.cpp
)

# Ativar AUTOMOC para projetos que usam Qt
set_target_properties(tests_qml PROPERTIES AUTOMOC ON)

# Adicionar diretórios de inclusão (depois de criar o executável)
target_include_directories(tests_qml PRIVATE ${Catch2_SOURCE_DIR}/src)

# Vincular dependências
target_link_libraries(tests_system
    PRIVATE
    Qt6::Core
    Qt6::Qml
    Qt6::Test
    # Catch2::Catch2
    Catch2::Catch2WithMain # <- Use esta opção para adicionar o main()
)


target_link_libraries(tests_zmqreader
    PRIVATE
    Qt6::Core
    Qt6::Qml
    Qt6::Test
    # Catch2::Catch2
    Catch2::Catch2WithMain
    zmq
)
# target_link_libraries(tests_zmqreader PRIVATE zmq) coloquei em cima

target_link_libraries(tests_qml
    PRIVATE
    Qt6::Core
    Qt6::Qml
    # Catch2::Catch2
    Catch2::Catch2WithMain # <- Use esta opção para adicionar o main()
)


# Adicionar ao CTest
enable_testing()
add_test(NAME tests_system COMMAND tests_system)
add_test(NAME tests_zmqreader COMMAND tests_zmqreader)
add_test(NAME tests_qml COMMAND tests_qml)