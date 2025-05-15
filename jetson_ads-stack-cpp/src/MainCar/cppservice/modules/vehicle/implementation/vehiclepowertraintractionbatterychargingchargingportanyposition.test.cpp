#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportanyposition.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition", "[VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition> testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition>();
    // setup your test
    SECTION("Test property SupportedInletTypes") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->getSupportedInletTypes() == Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC );
    }
    SECTION("Test property IsFlapOpen") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->setIsFlapOpen(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->getIsFlapOpen() == false );
    }
    SECTION("Test property IsChargingCableConnected") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->setIsChargingCableConnected(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->getIsChargingCableConnected() == false );
    }
    SECTION("Test property IsChargingCableLocked") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->setIsChargingCableLocked(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->getIsChargingCableLocked() == false );
    }
}
