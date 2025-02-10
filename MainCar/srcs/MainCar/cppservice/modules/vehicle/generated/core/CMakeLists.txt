set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)
set (SOURCES_CORE_SUPPORT
    vehicle.json.adapter.cpp
    vehicle.publisher.cpp
    vehicle.threadsafedecorator.cpp
    vehiclepowertrain.publisher.cpp
    vehiclepowertrain.threadsafedecorator.cpp
    vehiclepowertraintransmission.publisher.cpp
    vehiclepowertraintransmission.threadsafedecorator.cpp
    vehiclepowertrainelectricmotor.publisher.cpp
    vehiclepowertrainelectricmotor.threadsafedecorator.cpp
    vehiclepowertrainelectricmotorenginecoolant.publisher.cpp
    vehiclepowertrainelectricmotorenginecoolant.threadsafedecorator.cpp
    vehiclepowertraintractionbattery.publisher.cpp
    vehiclepowertraintractionbattery.threadsafedecorator.cpp
    vehiclepowertraintractionbatterytemperature.publisher.cpp
    vehiclepowertraintractionbatterytemperature.threadsafedecorator.cpp
    vehiclepowertraintractionbatterycellvoltage.publisher.cpp
    vehiclepowertraintractionbatterycellvoltage.threadsafedecorator.cpp
    vehiclepowertraintractionbatterystateofcharge.publisher.cpp
    vehiclepowertraintractionbatterystateofcharge.threadsafedecorator.cpp
    vehiclepowertraintractionbatterycharging.publisher.cpp
    vehiclepowertraintractionbatterycharging.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontleft.publisher.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontleft.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontmiddle.publisher.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontmiddle.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontright.publisher.cpp
    vehiclepowertraintractionbatterychargingchargingportfrontright.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargingportrearleft.publisher.cpp
    vehiclepowertraintractionbatterychargingchargingportrearleft.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargingportrearmiddle.publisher.cpp
    vehiclepowertraintractionbatterychargingchargingportrearmiddle.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargingportrearright.publisher.cpp
    vehiclepowertraintractionbatterychargingchargingportrearright.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargingportanyposition.publisher.cpp
    vehiclepowertraintractionbatterychargingchargingportanyposition.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingmaximumchargingcurrent.publisher.cpp
    vehiclepowertraintractionbatterychargingmaximumchargingcurrent.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargecurrent.publisher.cpp
    vehiclepowertraintractionbatterychargingchargecurrent.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingchargevoltage.publisher.cpp
    vehiclepowertraintractionbatterychargingchargevoltage.threadsafedecorator.cpp
    vehiclepowertraintractionbatterycharginglocation.publisher.cpp
    vehiclepowertraintractionbatterycharginglocation.threadsafedecorator.cpp
    vehiclepowertraintractionbatterychargingtimer.publisher.cpp
    vehiclepowertraintractionbatterychargingtimer.threadsafedecorator.cpp
    vehiclepowertraintractionbatterydcdc.publisher.cpp
    vehiclepowertraintractionbatterydcdc.threadsafedecorator.cpp
    vehiclepowertraintractionbatterybatteryconditioning.publisher.cpp
    vehiclepowertraintractionbatterybatteryconditioning.threadsafedecorator.cpp
    vehiclebody.publisher.cpp
    vehiclebody.threadsafedecorator.cpp
    vehiclebodyhorn.publisher.cpp
    vehiclebodyhorn.threadsafedecorator.cpp
    vehiclebodylights.publisher.cpp
    vehiclebodylights.threadsafedecorator.cpp
    vehiclebodylightsbeamlow.publisher.cpp
    vehiclebodylightsbeamlow.threadsafedecorator.cpp
    vehiclebodylightsbeamhigh.publisher.cpp
    vehiclebodylightsbeamhigh.threadsafedecorator.cpp
    vehiclebodylightsrunning.publisher.cpp
    vehiclebodylightsrunning.threadsafedecorator.cpp
    vehiclebodylightsbackup.publisher.cpp
    vehiclebodylightsbackup.threadsafedecorator.cpp
    vehiclebodylightsparking.publisher.cpp
    vehiclebodylightsparking.threadsafedecorator.cpp
    vehiclebodylightsfogrear.publisher.cpp
    vehiclebodylightsfogrear.threadsafedecorator.cpp
    vehiclebodylightsfogfront.publisher.cpp
    vehiclebodylightsfogfront.threadsafedecorator.cpp
    vehiclebodylightslicenseplate.publisher.cpp
    vehiclebodylightslicenseplate.threadsafedecorator.cpp
    vehiclebodylightsbrake.publisher.cpp
    vehiclebodylightsbrake.threadsafedecorator.cpp
    vehiclebodylightshazard.publisher.cpp
    vehiclebodylightshazard.threadsafedecorator.cpp
    vehiclebodylightsdirectionindicatorleft.publisher.cpp
    vehiclebodylightsdirectionindicatorleft.threadsafedecorator.cpp
    vehiclebodylightsdirectionindicatorright.publisher.cpp
    vehiclebodylightsdirectionindicatorright.threadsafedecorator.cpp
    vehicledriver.publisher.cpp
    vehicledriver.threadsafedecorator.cpp
    vehicledriveridentifierdeprecated.publisher.cpp
    vehicledriveridentifierdeprecated.threadsafedecorator.cpp
    vehicleconnectivity.publisher.cpp
    vehicleconnectivity.threadsafedecorator.cpp
    vehicleversionvss.publisher.cpp
    vehicleversionvss.threadsafedecorator.cpp
    vehiclevehicleidentification.publisher.cpp
    vehiclevehicleidentification.threadsafedecorator.cpp
    vehiclelowvoltagebattery.publisher.cpp
    vehiclelowvoltagebattery.threadsafedecorator.cpp
    vehicleacceleration.publisher.cpp
    vehicleacceleration.threadsafedecorator.cpp
    vehicleangularvelocity.publisher.cpp
    vehicleangularvelocity.threadsafedecorator.cpp
    vehiclecurrentlocation.publisher.cpp
    vehiclecurrentlocation.threadsafedecorator.cpp
    vehiclecurrentlocationgnssreceiver.publisher.cpp
    vehiclecurrentlocationgnssreceiver.threadsafedecorator.cpp
    vehiclecurrentlocationgnssreceivermountingposition.publisher.cpp
    vehiclecurrentlocationgnssreceivermountingposition.threadsafedecorator.cpp
    vehiclediagnostics.publisher.cpp
    vehiclediagnostics.threadsafedecorator.cpp
    vehiclemotionmanagementelectricaxlerow1.publisher.cpp
    vehiclemotionmanagementelectricaxlerow1.threadsafedecorator.cpp
    vehiclemotionmanagementelectricaxlerow2.publisher.cpp
    vehiclemotionmanagementelectricaxlerow2.threadsafedecorator.cpp
    vehiclemotionmanagementbrake.publisher.cpp
    vehiclemotionmanagementbrake.threadsafedecorator.cpp
    vehiclemotionmanagementbrakeaxlerow1.publisher.cpp
    vehiclemotionmanagementbrakeaxlerow1.threadsafedecorator.cpp
    vehiclemotionmanagementbrakeaxlerow1wheelleft.publisher.cpp
    vehiclemotionmanagementbrakeaxlerow1wheelleft.threadsafedecorator.cpp
    vehiclemotionmanagementbrakeaxlerow1wheelright.publisher.cpp
    vehiclemotionmanagementbrakeaxlerow1wheelright.threadsafedecorator.cpp
    vehiclemotionmanagementbrakeaxlerow2.publisher.cpp
    vehiclemotionmanagementbrakeaxlerow2.threadsafedecorator.cpp
    vehiclemotionmanagementbrakeaxlerow2wheelleft.publisher.cpp
    vehiclemotionmanagementbrakeaxlerow2wheelleft.threadsafedecorator.cpp
    vehiclemotionmanagementbrakeaxlerow2wheelright.publisher.cpp
    vehiclemotionmanagementbrakeaxlerow2wheelright.threadsafedecorator.cpp
    vehiclemotionmanagementsteeringsteeringwheel.publisher.cpp
    vehiclemotionmanagementsteeringsteeringwheel.threadsafedecorator.cpp
    vehiclemotionmanagementsteeringaxlerow1.publisher.cpp
    vehiclemotionmanagementsteeringaxlerow1.threadsafedecorator.cpp
)
add_library(vehicle-core SHARED ${SOURCES_CORE_SUPPORT})
add_library(vehicle::vehicle-core ALIAS vehicle-core)
target_include_directories(vehicle-core
    PUBLIC
    $<BUILD_INTERFACE:${MODULES_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vehicle-core PUBLIC vehicle::vehicle-api nlohmann_json::nlohmann_json)
# ensure maximum compiler support
if(NOT MSVC)
  target_compile_options(vehicle-core PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden)
else()
  target_compile_options(vehicle-core PRIVATE /W4 /WX /wd4251)
endif()

install(TARGETS vehicle-core
        EXPORT VehicleCoreTargets)
# install includes
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/vehicle/generated FILES_MATCHING PATTERN "*.h")

export(EXPORT VehicleCoreTargets
  NAMESPACE vehicle::
)

install(EXPORT VehicleCoreTargets
  FILE VehicleCoreTargets.cmake
  DESTINATION ${InstallDir}
  NAMESPACE vehicle::
)
