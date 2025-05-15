#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodylightsdirectionindicatorleft.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyLightsDirectionIndicatorLeft", "[VehicleBodyLightsDirectionIndicatorLeft]"){
    std::unique_ptr<IVehicleBodyLightsDirectionIndicatorLeft> testVehicleBodyLightsDirectionIndicatorLeft = std::make_unique<VehicleBodyLightsDirectionIndicatorLeft>();
    // setup your test
    SECTION("Test property IsSignaling") {
        // Do implement test here
        testVehicleBodyLightsDirectionIndicatorLeft->setIsSignaling(false);
        REQUIRE( testVehicleBodyLightsDirectionIndicatorLeft->getIsSignaling() == false );
    }
    SECTION("Test property IsDefect") {
        // Do implement test here
        testVehicleBodyLightsDirectionIndicatorLeft->setIsDefect(false);
        REQUIRE( testVehicleBodyLightsDirectionIndicatorLeft->getIsDefect() == false );
    }
}
