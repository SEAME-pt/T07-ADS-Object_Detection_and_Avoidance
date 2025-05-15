#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportrearmiddle.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle", "[VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle> testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle>();
    // setup your test
    SECTION("Test property SupportedInletTypes") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle->getSupportedInletTypes() == Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC );
    }
    SECTION("Test property IsFlapOpen") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle->setIsFlapOpen(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle->getIsFlapOpen() == false );
    }
    SECTION("Test property IsChargingCableConnected") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle->setIsChargingCableConnected(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle->getIsChargingCableConnected() == false );
    }
    SECTION("Test property IsChargingCableLocked") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle->setIsChargingCableLocked(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle->getIsChargingCableLocked() == false );
    }
}
