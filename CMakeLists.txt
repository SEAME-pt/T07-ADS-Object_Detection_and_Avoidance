cmake_minimum_required(VERSION 3.10)
project(JetRacerMPC)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
find_package(CppAD REQUIRED)
find_library(IPOPT_LIBRARY NAMES ipopt)

add_executable(jetracer_mpc jetracer_mpc.cpp)
target_include_directories(jetracer_mpc PRIVATE ${EIGEN3_INCLUDE_DIR} ${CppAD_INCLUDE_DIRS})
target_link_libraries(jetracer_mpc ${IPOPT_LIBRARY} cppad_lib)