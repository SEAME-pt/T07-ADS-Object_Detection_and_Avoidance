set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (SOURCES
    datastructs.api.cpp
)
add_library(vehicle-api SHARED ${SOURCES})
add_library(vehicle::vehicle-api ALIAS vehicle-api)
target_include_directories(vehicle-api
    PUBLIC
    $<BUILD_INTERFACE:${MODULES_DIR}>
    $<INSTALL_INTERFACE:include>
)
# ensure maximum compiler support
if(NOT MSVC)
  target_compile_options(vehicle-api PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden)
else()
  target_compile_options(vehicle-api PRIVATE /W4 /WX /wd4251)
endif()

# install binary files
install(TARGETS vehicle-api
        EXPORT VehicleApiTargets)
# install includes
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/vehicle/generated FILES_MATCHING PATTERN "*.h")

export(EXPORT VehicleApiTargets
  NAMESPACE vehicle::
)

install(EXPORT VehicleApiTargets
  FILE VehicleApiTargets.cmake
  DESTINATION ${InstallDir}
  NAMESPACE vehicle::
)
