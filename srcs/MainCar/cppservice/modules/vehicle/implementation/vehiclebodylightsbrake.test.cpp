#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsbrake.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsBrake", "[VehicleBodyLightsBrake]"){
    std::unique_ptr<IVehicleBodyLightsBrake> testVehicleBodyLightsBrake = std::make_unique<VehicleBodyLightsBrake>();
    // setup your test
    SECTION("Test property IsActive") {
        // Do implement test here
        testVehicleBodyLightsBrake->setIsActive(Vehicle_Body_Lights_Brake_IsActive_ValueEnum::INACTIVE);
        REQUIRE( testVehicleBodyLightsBrake->getIsActive() == Vehicle_Body_Lights_Brake_IsActive_ValueEnum::INACTIVE );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsBrake->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsBrake->getIsDefect() == false );
    }
}
