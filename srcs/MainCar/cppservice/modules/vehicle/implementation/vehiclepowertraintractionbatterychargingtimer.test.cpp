#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingtimer.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingTimer", "[VehiclePowertrainTractionBatteryChargingTimer]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingTimer> testVehiclePowertrainTractionBatteryChargingTimer = std::make_unique<VehiclePowertrainTractionBatteryChargingTimer>();
    // setup your test
    SECTION("Test property Mode") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingTimer->setMode(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum::INACTIVE);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingTimer->getMode() == Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum::INACTIVE );
    }
    SECTION("Test property Time") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingTimer->setTime(std::string());
        REQUIRE( testVehiclePowertrainTractionBatteryChargingTimer->getTime() == std::string() );
    }
}
