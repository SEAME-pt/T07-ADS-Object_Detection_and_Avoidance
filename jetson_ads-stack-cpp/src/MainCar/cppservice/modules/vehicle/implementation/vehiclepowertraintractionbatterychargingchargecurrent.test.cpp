#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargecurrent.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingChargeCurrent", "[VehiclePowertrainTractionBatteryChargingChargeCurrent]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargeCurrent> testVehiclePowertrainTractionBatteryChargingChargeCurrent = std::make_unique<VehiclePowertrainTractionBatteryChargingChargeCurrent>();
    // setup your test
    SECTION("Test property DC") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargeCurrent->setDc(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargeCurrent->getDc() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase1") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargeCurrent->setPhase1(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargeCurrent->getPhase1() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase2") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargeCurrent->setPhase2(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargeCurrent->getPhase2() == Approx( 0.0f ) );
    }
    SECTION("Test property Phase3") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingChargeCurrent->setPhase3(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingChargeCurrent->getPhase3() == Approx( 0.0f ) );
    }
}
