#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsfogfront.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsFogFront", "[VehicleBodyLightsFogFront]"){
    std::unique_ptr<IVehicleBodyLightsFogFront> testVehicleBodyLightsFogFront = std::make_unique<VehicleBodyLightsFogFront>();
    // setup your test
    SECTION("Test property IsOn") {
        // Do implement test here
        testVehicleBodyLightsFogFront->setIsOn(false);
        REQUIRE( testVehicleBodyLightsFogFront->getIsOn() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsFogFront->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsFogFront->getIsDefect() == false );
    }
}
