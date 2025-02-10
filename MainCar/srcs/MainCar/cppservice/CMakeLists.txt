cmake_minimum_required(VERSION 3.24)
project(cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# on Windows it is helpful to have all binary files next to each other
# it is intentionally not set as part of BUILD_TESTING to have a consistent behavior
# this can be removed once there is a better option than modifying the PATH env for unit testing
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_TESTING "Enable this option to build the test targets" OFF)
option(CPP_FETCH_DEPS "Fetch and build dependencies, if not pre-installed" OFF)

# enable testing
if(BUILD_TESTING)
include(CTest)
enable_testing()
endif(BUILD_TESTING)

if(CPP_FETCH_DEPS)
include(FetchContent)

if(BUILD_TESTING)
find_package(Catch2 QUIET)
if (NOT Catch2_Found)
  message(STATUS "Catch2 NOT FOUND, fetching the git repository")
  FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v2.13.8
      GIT_SHALLOW    TRUE
      OVERRIDE_FIND_PACKAGE)
  FetchContent_MakeAvailable(Catch2)
endif()

find_package(trompeloeil QUIET)
if(NOT trompeloeil_Found)
  message(STATUS "trompeloeil NOT FOUND, fetching the git repository")
  FetchContent_Declare(
      trompeloeil
      GIT_REPOSITORY https://github.com/rollbear/trompeloeil.git
      GIT_TAG        v46
      OVERRIDE_FIND_PACKAGE
      )
  FetchContent_MakeAvailable(trompeloeil)
endif()
endif() # BUILD_TESTING

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  # pull nlohmann json as dependency
  message(STATUS "nlohmann_json NOT FOUND, fetching the release package")
  include(FetchContent)
  set(JSON_Install ON)
  FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  OVERRIDE_FIND_PACKAGE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(olink_core QUIET)
if(NOT olink_core_FOUND)
  # pull olink_core as dependency
  message(STATUS "olink_core NOT FOUND, fetching the git repository")
  FetchContent_Declare(olink_core
      GIT_REPOSITORY https://github.com/apigear-io/objectlink-core-cpp.git
      GIT_TAG v0.2.10
      GIT_SHALLOW TRUE
      EXCLUDE_FROM_ALL FALSE
      OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(olink_core)
endif()
endif() # CPP_FETCH_DEPS

# checks are workaround until generator support feature check properly
find_package(vehicle QUIET)
if(NOT vehicle_FOUND)
  # pull vehicle as dependency
  message(STATUS "vehicle NOT FOUND, building from source folder")
  include(FetchContent)
  FetchContent_Declare(vehicle
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/vehicle"
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(vehicle)
endif()
