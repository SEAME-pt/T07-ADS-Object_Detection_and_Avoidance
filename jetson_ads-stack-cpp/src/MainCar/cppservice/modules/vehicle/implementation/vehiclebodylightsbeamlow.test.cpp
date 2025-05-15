#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsbeamlow.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsBeamLow", "[VehicleBodyLightsBeamLow]"){
    std::unique_ptr<IVehicleBodyLightsBeamLow> testVehicleBodyLightsBeamLow = std::make_unique<VehicleBodyLightsBeamLow>();
    // setup your test
    SECTION("Test property IsOn") {
        // Do implement test here
        testVehicleBodyLightsBeamLow->setIsOn(false);
        REQUIRE( testVehicleBodyLightsBeamLow->getIsOn() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsBeamLow->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsBeamLow->getIsDefect() == false );
    }
}
