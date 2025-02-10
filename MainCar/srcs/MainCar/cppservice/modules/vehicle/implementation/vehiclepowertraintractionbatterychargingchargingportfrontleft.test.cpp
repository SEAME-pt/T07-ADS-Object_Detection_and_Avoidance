#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontleft.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft", "[VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft> testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft>();
    // setup your test
    SECTION("Test property SupportedInletTypes") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft->getSupportedInletTypes() == Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC );
    }
    SECTION("Test property IsFlapOpen") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft->setIsFlapOpen(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft->getIsFlapOpen() == false );
    }
    SECTION("Test property IsChargingCableConnected") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft->setIsChargingCableConnected(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft->getIsChargingCableConnected() == false );
    }
    SECTION("Test property IsChargingCableLocked") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft->setIsChargingCableLocked(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft->getIsChargingCableLocked() == false );
    }
}
