#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsbackup.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsBackup", "[VehicleBodyLightsBackup]"){
    std::unique_ptr<IVehicleBodyLightsBackup> testVehicleBodyLightsBackup = std::make_unique<VehicleBodyLightsBackup>();
    // setup your test
    SECTION("Test property IsOn") {
        // Do implement test here
        testVehicleBodyLightsBackup->setIsOn(false);
        REQUIRE( testVehicleBodyLightsBackup->getIsOn() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsBackup->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsBackup->getIsDefect() == false );
    }
}
