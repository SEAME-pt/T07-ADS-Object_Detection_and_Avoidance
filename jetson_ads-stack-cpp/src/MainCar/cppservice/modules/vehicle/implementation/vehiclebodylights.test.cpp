#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylights.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLights", "[VehicleBodyLights]"){
    std::unique_ptr<IVehicleBodyLights> testVehicleBodyLights = std::make_unique<VehicleBodyLights>();
    // setup your test
    SECTION("Test property LightSwitch") {
        // Do implement test here
        testVehicleBodyLights->setLightSwitch(Vehicle_Body_Lights_LightSwitch_ValueEnum::OFF);
        REQUIRE( testVehicleBodyLights->getLightSwitch() == Vehicle_Body_Lights_LightSwitch_ValueEnum::OFF );
    }
    SECTION("Test property IsHighBeamSwitchOn") {
        // Do implement test here
        testVehicleBodyLights->setIsHighBeamSwitchOn(false);
        REQUIRE( testVehicleBodyLights->getIsHighBeamSwitchOn() == false );
    }
}
