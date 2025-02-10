#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsparking.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsParking", "[VehicleBodyLightsParking]"){
    std::unique_ptr<IVehicleBodyLightsParking> testVehicleBodyLightsParking = std::make_unique<VehicleBodyLightsParking>();
    // setup your test
    SECTION("Test property IsOn") {
        // Do implement test here
        testVehicleBodyLightsParking->setIsOn(false);
        REQUIRE( testVehicleBodyLightsParking->getIsOn() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsParking->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsParking->getIsDefect() == false );
    }
}
