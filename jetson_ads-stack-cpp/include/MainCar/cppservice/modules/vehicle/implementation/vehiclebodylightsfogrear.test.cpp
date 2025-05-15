#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsfogrear.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsFogRear", "[VehicleBodyLightsFogRear]"){
    std::unique_ptr<IVehicleBodyLightsFogRear> testVehicleBodyLightsFogRear = std::make_unique<VehicleBodyLightsFogRear>();
    // setup your test
    SECTION("Test property IsOn") {
        // Do implement test here
        testVehicleBodyLightsFogRear->setIsOn(false);
        REQUIRE( testVehicleBodyLightsFogRear->getIsOn() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsFogRear->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsFogRear->getIsDefect() == false );
    }
}
