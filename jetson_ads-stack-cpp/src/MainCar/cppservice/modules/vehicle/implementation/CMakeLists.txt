set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set (SOURCES_CORE_IMPL
    vehicle.cpp
    vehiclepowertrain.cpp
    vehiclepowertraintransmission.cpp
    vehiclepowertrainelectricmotor.cpp
    vehiclepowertrainelectricmotorenginecoolant.cpp
    vehiclepowertraintractionbattery.cpp
    vehiclepowertraintractionbatterytemperature.cpp
    vehiclepowertraintractionbatterycellvoltage.cpp
    vehiclepowertraintractionbatterystateofcharge.cpp
    vehiclepowertraintractionbatterycharging.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontleft.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontmiddle.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontright.cpp
    vehiclepowertraintractionbatterychargingchargingportrearleft.cpp
    vehiclepowertraintractionbatterychargingchargingportrearmiddle.cpp
    vehiclepowertraintractionbatterychargingchargingportrearright.cpp
    vehiclepowertraintractionbatterychargingchargingportanyposition.cpp
    vehiclepowertraintractionbatterychargingmaximumchargingcurrent.cpp
    vehiclepowertraintractionbatterychargingchargecurrent.cpp
    vehiclepowertraintractionbatterychargingchargevoltage.cpp
    vehiclepowertraintractionbatterycharginglocation.cpp
    vehiclepowertraintractionbatterychargingtimer.cpp
    vehiclepowertraintractionbatterydcdc.cpp
    vehiclepowertraintractionbatterybatteryconditioning.cpp
    vehiclebody.cpp
    vehiclebodyhorn.cpp
    vehiclebodylights.cpp
    vehiclebodylightsbeamlow.cpp
    vehiclebodylightsbeamhigh.cpp
    vehiclebodylightsrunning.cpp
    vehiclebodylightsbackup.cpp
    vehiclebodylightsparking.cpp
    vehiclebodylightsfogrear.cpp
    vehiclebodylightsfogfront.cpp
    vehiclebodylightslicenseplate.cpp
    vehiclebodylightsbrake.cpp
    vehiclebodylightshazard.cpp
    vehiclebodylightsdirectionindicatorleft.cpp
    vehiclebodylightsdirectionindicatorright.cpp
    vehicledriver.cpp
    vehicledriveridentifierdeprecated.cpp
    vehicleconnectivity.cpp
    vehicleversionvss.cpp
    vehiclevehicleidentification.cpp
    vehiclelowvoltagebattery.cpp
    vehicleacceleration.cpp
    vehicleangularvelocity.cpp
    vehiclecurrentlocation.cpp
    vehiclecurrentlocationgnssreceiver.cpp
    vehiclecurrentlocationgnssreceivermountingposition.cpp
    vehiclediagnostics.cpp
    vehiclemotionmanagementelectricaxlerow1.cpp
    vehiclemotionmanagementelectricaxlerow2.cpp
    vehiclemotionmanagementbrake.cpp
    vehiclemotionmanagementbrakeaxlerow1.cpp
    vehiclemotionmanagementbrakeaxlerow1wheelleft.cpp
    vehiclemotionmanagementbrakeaxlerow1wheelright.cpp
    vehiclemotionmanagementbrakeaxlerow2.cpp
    vehiclemotionmanagementbrakeaxlerow2wheelleft.cpp
    vehiclemotionmanagementbrakeaxlerow2wheelright.cpp
    vehiclemotionmanagementsteeringsteeringwheel.cpp
    vehiclemotionmanagementsteeringaxlerow1.cpp
)
add_library(vehicle-implementation SHARED ${SOURCES_CORE_IMPL})
add_library(vehicle::vehicle-implementation ALIAS vehicle-implementation)
target_include_directories(vehicle-implementation
    PUBLIC
    $<BUILD_INTERFACE:${MODULES_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vehicle-implementation PUBLIC vehicle::vehicle-api PRIVATE vehicle::vehicle-core Threads::Threads)
# ensure maximum compiler support
if(NOT MSVC)
  target_compile_options(vehicle-implementation PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden)
else()
  target_compile_options(vehicle-implementation PRIVATE /W4 /WX /wd4251)
endif()

# add test cases
if(BUILD_TESTING)

find_package(Catch2 REQUIRED)

set (SOURCES_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/../generated/core/vehicle.test.cpp
    vehicle.test.cpp
    vehiclepowertrain.test.cpp
    vehiclepowertraintransmission.test.cpp
    vehiclepowertrainelectricmotor.test.cpp
    vehiclepowertrainelectricmotorenginecoolant.test.cpp
    vehiclepowertraintractionbattery.test.cpp
    vehiclepowertraintractionbatterytemperature.test.cpp
    vehiclepowertraintractionbatterycellvoltage.test.cpp
    vehiclepowertraintractionbatterystateofcharge.test.cpp
    vehiclepowertraintractionbatterycharging.test.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontleft.test.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontmiddle.test.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontright.test.cpp
    vehiclepowertraintractionbatterychargingchargingportrearleft.test.cpp
    vehiclepowertraintractionbatterychargingchargingportrearmiddle.test.cpp
    vehiclepowertraintractionbatterychargingchargingportrearright.test.cpp
    vehiclepowertraintractionbatterychargingchargingportanyposition.test.cpp
    vehiclepowertraintractionbatterychargingmaximumchargingcurrent.test.cpp
    vehiclepowertraintractionbatterychargingchargecurrent.test.cpp
    vehiclepowertraintractionbatterychargingchargevoltage.test.cpp
    vehiclepowertraintractionbatterycharginglocation.test.cpp
    vehiclepowertraintractionbatterychargingtimer.test.cpp
    vehiclepowertraintractionbatterydcdc.test.cpp
    vehiclepowertraintractionbatterybatteryconditioning.test.cpp
    vehiclebody.test.cpp
    vehiclebodyhorn.test.cpp
    vehiclebodylights.test.cpp
    vehiclebodylightsbeamlow.test.cpp
    vehiclebodylightsbeamhigh.test.cpp
    vehiclebodylightsrunning.test.cpp
    vehiclebodylightsbackup.test.cpp
    vehiclebodylightsparking.test.cpp
    vehiclebodylightsfogrear.test.cpp
    vehiclebodylightsfogfront.test.cpp
    vehiclebodylightslicenseplate.test.cpp
    vehiclebodylightsbrake.test.cpp
    vehiclebodylightshazard.test.cpp
    vehiclebodylightsdirectionindicatorleft.test.cpp
    vehiclebodylightsdirectionindicatorright.test.cpp
    vehicledriver.test.cpp
    vehicledriveridentifierdeprecated.test.cpp
    vehicleconnectivity.test.cpp
    vehicleversionvss.test.cpp
    vehiclevehicleidentification.test.cpp
    vehiclelowvoltagebattery.test.cpp
    vehicleacceleration.test.cpp
    vehicleangularvelocity.test.cpp
    vehiclecurrentlocation.test.cpp
    vehiclecurrentlocationgnssreceiver.test.cpp
    vehiclecurrentlocationgnssreceivermountingposition.test.cpp
    vehiclediagnostics.test.cpp
    vehiclemotionmanagementelectricaxlerow1.test.cpp
    vehiclemotionmanagementelectricaxlerow2.test.cpp
    vehiclemotionmanagementbrake.test.cpp
    vehiclemotionmanagementbrakeaxlerow1.test.cpp
    vehiclemotionmanagementbrakeaxlerow1wheelleft.test.cpp
    vehiclemotionmanagementbrakeaxlerow1wheelright.test.cpp
    vehiclemotionmanagementbrakeaxlerow2.test.cpp
    vehiclemotionmanagementbrakeaxlerow2wheelleft.test.cpp
    vehiclemotionmanagementbrakeaxlerow2wheelright.test.cpp
    vehiclemotionmanagementsteeringsteeringwheel.test.cpp
    vehiclemotionmanagementsteeringaxlerow1.test.cpp
)
add_executable(test_vehicle
    ${SOURCES_TEST}
)
target_link_libraries(test_vehicle vehicle::vehicle-implementation Catch2::Catch2)
target_include_directories(test_vehicle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ensure maximum compiler support
if(NOT MSVC)
  target_compile_options(test_vehicle PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden)
else()
  target_compile_options(test_vehicle PRIVATE /W4 /WX /wd4251)
endif()

add_test(NAME test_vehicle COMMAND $<TARGET_FILE:test_vehicle>)
endif(BUILD_TESTING)

install(TARGETS vehicle-implementation
        EXPORT VehicleImplementationTargets)
# install includes
FILE(GLOB Vehicle_INCLUDES *.h)
install(FILES ${Vehicle_INCLUDES}
        DESTINATION include/vehicle/implementation)

export(EXPORT VehicleImplementationTargets
  NAMESPACE vehicle::
)

install(EXPORT VehicleImplementationTargets
  FILE VehicleImplementationTargets.cmake
  DESTINATION ${InstallDir}
  NAMESPACE vehicle::
)
