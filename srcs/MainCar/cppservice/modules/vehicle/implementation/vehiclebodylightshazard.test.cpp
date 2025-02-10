#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightshazard.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsHazard", "[VehicleBodyLightsHazard]"){
    std::unique_ptr<IVehicleBodyLightsHazard> testVehicleBodyLightsHazard = std::make_unique<VehicleBodyLightsHazard>();
    // setup your test
    SECTION("Test property IsSignaling") {
        // Do implement test here
        testVehicleBodyLightsHazard->setIsSignaling(false);
        REQUIRE( testVehicleBodyLightsHazard->getIsSignaling() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsHazard->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsHazard->getIsDefect() == false );
    }
}
