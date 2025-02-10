#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportrearright.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingChargingPortRearRight", "[VehiclePowertrainTractionBatteryChargingChargingPortRearRight]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearRight> testVehiclePowertrainTractionBatteryChargingChargingPortRearRight = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortRearRight>();
    // setup your test
    SECTION("Test property SupportedInletTypes") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->getSupportedInletTypes() == Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC );
    }
    SECTION("Test property IsFlapOpen") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->setIsFlapOpen(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->getIsFlapOpen() == false );
    }
    SECTION("Test property IsChargingCableConnected") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->setIsChargingCableConnected(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->getIsChargingCableConnected() == false );
    }
    SECTION("Test property IsChargingCableLocked") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->setIsChargingCableLocked(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->getIsChargingCableLocked() == false );
    }
}
