#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterybatteryconditioning.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryBatteryConditioning", "[VehiclePowertrainTractionBatteryBatteryConditioning]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryBatteryConditioning> testVehiclePowertrainTractionBatteryBatteryConditioning = std::make_unique<VehiclePowertrainTractionBatteryBatteryConditioning>();
    // setup your test
    SECTION("Test property IsActive") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryBatteryConditioning->setIsActive(false);
        REQUIRE( testVehiclePowertrainTractionBatteryBatteryConditioning->getIsActive() == false );
    }
    SECTION("Test property IsOngoing") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryBatteryConditioning->setIsOngoing(false);
        REQUIRE( testVehiclePowertrainTractionBatteryBatteryConditioning->getIsOngoing() == false );
    }
    SECTION("Test property RequestedMode") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryBatteryConditioning->setRequestedMode(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum::INACTIVE);
        REQUIRE( testVehiclePowertrainTractionBatteryBatteryConditioning->getRequestedMode() == Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum::INACTIVE );
    }
    SECTION("Test property StartTime") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryBatteryConditioning->setStartTime(std::string());
        REQUIRE( testVehiclePowertrainTractionBatteryBatteryConditioning->getStartTime() == std::string() );
    }
    SECTION("Test property TargetTime") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryBatteryConditioning->setTargetTime(std::string());
        REQUIRE( testVehiclePowertrainTractionBatteryBatteryConditioning->getTargetTime() == std::string() );
    }
    SECTION("Test property TargetTemperature") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryBatteryConditioning->setTargetTemperature(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryBatteryConditioning->getTargetTemperature() == Approx( 0.0f ) );
    }
}
