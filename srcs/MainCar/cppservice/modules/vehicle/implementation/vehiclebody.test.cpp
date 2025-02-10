#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebody.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBody", "[VehicleBody]"){
    std::unique_ptr<IVehicleBody> testVehicleBody = std::make_unique<VehicleBody>();
    // setup your test
    SECTION("Test property PowerOptimizeLevel") {
        // Do implement test here
        testVehicleBody->setPowerOptimizeLevel(0);
        REQUIRE( testVehicleBody->getPowerOptimizeLevel() == 0 );
    }
    SECTION("Test property IsAutoPowerOptimize") {
        // Do implement test here
        testVehicleBody->setIsAutoPowerOptimize(false);
        REQUIRE( testVehicleBody->getIsAutoPowerOptimize() == false );
    }
}
