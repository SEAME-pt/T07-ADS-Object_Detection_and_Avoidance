#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontright.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingChargingPortFrontRight", "[VehiclePowertrainTractionBatteryChargingChargingPortFrontRight]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight> testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortFrontRight>();
    // setup your test
    SECTION("Test property SupportedInletTypes") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight->getSupportedInletTypes() == Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC );
    }
    SECTION("Test property IsFlapOpen") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight->setIsFlapOpen(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight->getIsFlapOpen() == false );
    }
    SECTION("Test property IsChargingCableConnected") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight->setIsChargingCableConnected(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight->getIsChargingCableConnected() == false );
    }
    SECTION("Test property IsChargingCableLocked") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight->setIsChargingCableLocked(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontRight->getIsChargingCableLocked() == false );
    }
}
