#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsdirectionindicatorright.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsDirectionIndicatorRight", "[VehicleBodyLightsDirectionIndicatorRight]"){
    std::unique_ptr<IVehicleBodyLightsDirectionIndicatorRight> testVehicleBodyLightsDirectionIndicatorRight = std::make_unique<VehicleBodyLightsDirectionIndicatorRight>();
    // setup your test
    SECTION("Test property IsSignaling") {
        // Do implement test here
        testVehicleBodyLightsDirectionIndicatorRight->setIsSignaling(false);
        REQUIRE( testVehicleBodyLightsDirectionIndicatorRight->getIsSignaling() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsDirectionIndicatorRight->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsDirectionIndicatorRight->getIsDefect() == false );
    }
}
