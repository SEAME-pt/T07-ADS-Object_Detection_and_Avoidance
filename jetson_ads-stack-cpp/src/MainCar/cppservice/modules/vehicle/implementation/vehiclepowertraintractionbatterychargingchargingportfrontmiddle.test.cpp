#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontmiddle.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle", "[VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle> testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle>();
    // setup your test
    SECTION("Test property SupportedInletTypes") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle->getSupportedInletTypes() == Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC );
    }
    SECTION("Test property IsFlapOpen") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle->setIsFlapOpen(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle->getIsFlapOpen() == false );
    }
    SECTION("Test property IsChargingCableConnected") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle->setIsChargingCableConnected(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle->getIsChargingCableConnected() == false );
    }
    SECTION("Test property IsChargingCableLocked") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle->setIsChargingCableLocked(false);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle->getIsChargingCableLocked() == false );
    }
}
