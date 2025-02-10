#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsrunning.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsRunning", "[VehicleBodyLightsRunning]"){
    std::unique_ptr<IVehicleBodyLightsRunning> testVehicleBodyLightsRunning = std::make_unique<VehicleBodyLightsRunning>();
    // setup your test
    SECTION("Test property IsOn") {
        // Do implement test here
        testVehicleBodyLightsRunning->setIsOn(false);
        REQUIRE( testVehicleBodyLightsRunning->getIsOn() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsRunning->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsRunning->getIsDefect() == false );
    }
}
