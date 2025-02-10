#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent", "[VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent> testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent = std::make_unique<VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent>();
    // setup your test
    SECTION("Test property DC") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent->setDc(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent->getDc() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase1") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent->setPhase1(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent->getPhase1() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase2") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent->setPhase2(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent->getPhase2() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase3") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent->setPhase3(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent->getPhase3() == Approx( 0.0f ) );
    }
}
