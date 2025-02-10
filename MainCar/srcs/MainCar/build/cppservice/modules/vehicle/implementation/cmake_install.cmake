# Install script for directory: /workspace/cppservice/modules/vehicle/implementation

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libvehicle-implementation.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libvehicle-implementation.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libvehicle-implementation.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/workspace/build/cppservice/modules/vehicle/implementation/libvehicle-implementation.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libvehicle-implementation.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libvehicle-implementation.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libvehicle-implementation.so"
         OLD_RPATH "/workspace/build/cppservice/modules/vehicle/generated/core:/workspace/build/cppservice/modules/vehicle/generated/api:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libvehicle-implementation.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/vehicle/implementation" TYPE FILE FILES
    "/workspace/cppservice/modules/vehicle/implementation/vehicle.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehicleacceleration.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehicleangularvelocity.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebody.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodyhorn.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylights.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsbackup.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsbeamhigh.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsbeamlow.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsbrake.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsdirectionindicatorleft.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsdirectionindicatorright.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsfogfront.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsfogrear.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightshazard.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightslicenseplate.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsparking.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclebodylightsrunning.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehicleconnectivity.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclecurrentlocation.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclecurrentlocationgnssreceiver.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclecurrentlocationgnssreceivermountingposition.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclediagnostics.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehicledriver.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehicledriveridentifierdeprecated.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclelowvoltagebattery.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementbrake.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1wheelleft.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1wheelright.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2wheelleft.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2wheelright.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementelectricaxlerow1.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementelectricaxlerow2.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementsteeringaxlerow1.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclemotionmanagementsteeringsteeringwheel.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertrain.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertrainelectricmotor.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertrainelectricmotorenginecoolant.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbattery.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterybatteryconditioning.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterycellvoltage.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterycharging.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargecurrent.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargevoltage.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportanyposition.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontleft.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontmiddle.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontright.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportrearleft.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportrearmiddle.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportrearright.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterycharginglocation.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterychargingtimer.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterydcdc.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterystateofcharge.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintractionbatterytemperature.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclepowertraintransmission.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehiclevehicleidentification.h"
    "/workspace/cppservice/modules/vehicle/implementation/vehicleversionvss.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/vehicle/VehicleImplementationTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/vehicle/VehicleImplementationTargets.cmake"
         "/workspace/build/cppservice/modules/vehicle/implementation/CMakeFiles/Export/cf420b7bf9a00283d2843121b8c0cb36/VehicleImplementationTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/vehicle/VehicleImplementationTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/vehicle/VehicleImplementationTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/vehicle" TYPE FILE FILES "/workspace/build/cppservice/modules/vehicle/implementation/CMakeFiles/Export/cf420b7bf9a00283d2843121b8c0cb36/VehicleImplementationTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/vehicle" TYPE FILE FILES "/workspace/build/cppservice/modules/vehicle/implementation/CMakeFiles/Export/cf420b7bf9a00283d2843121b8c0cb36/VehicleImplementationTargets-noconfig.cmake")
  endif()
endif()

