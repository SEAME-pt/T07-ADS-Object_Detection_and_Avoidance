#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightslicenseplate.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsLicensePlate", "[VehicleBodyLightsLicensePlate]"){
    std::unique_ptr<IVehicleBodyLightsLicensePlate> testVehicleBodyLightsLicensePlate = std::make_unique<VehicleBodyLightsLicensePlate>();
    // setup your test
    SECTION("Test property IsOn") {
        // Do implement test here
        testVehicleBodyLightsLicensePlate->setIsOn(false);
        REQUIRE( testVehicleBodyLightsLicensePlate->getIsOn() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsLicensePlate->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsLicensePlate->getIsDefect() == false );
    }
}
