#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsbeamhigh.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsBeamHigh", "[VehicleBodyLightsBeamHigh]"){
    std::unique_ptr<IVehicleBodyLightsBeamHigh> testVehicleBodyLightsBeamHigh = std::make_unique<VehicleBodyLightsBeamHigh>();
    // setup your test
    SECTION("Test property IsOn") {
        // Do implement test here
        testVehicleBodyLightsBeamHigh->setIsOn(false);
        REQUIRE( testVehicleBodyLightsBeamHigh->getIsOn() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsBeamHigh->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsBeamHigh->getIsDefect() == false );
    }
}
