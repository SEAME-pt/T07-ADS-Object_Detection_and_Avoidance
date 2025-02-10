#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargevoltage.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingChargeVoltage", "[VehiclePowertrainTractionBatteryChargingChargeVoltage]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargeVoltage> testVehiclePowertrainTractionBatteryChargingChargeVoltage = std::make_unique<VehiclePowertrainTractionBatteryChargingChargeVoltage>();
    // setup your test
    SECTION("Test property DC") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargeVoltage->setDc(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargeVoltage->getDc() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase1") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargeVoltage->setPhase1(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargeVoltage->getPhase1() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase2") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargeVoltage->setPhase2(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargeVoltage->getPhase2() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase3") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargeVoltage->setPhase3(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargeVoltage->getPhase3() == Approx( 0.0f ) );
    }
}
