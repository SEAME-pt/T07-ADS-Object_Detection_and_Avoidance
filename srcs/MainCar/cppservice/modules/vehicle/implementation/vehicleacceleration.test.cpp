#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehicleacceleration.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleAcceleration", "[VehicleAcceleration]"){
    std::unique_ptr<IVehicleAcceleration> testVehicleAcceleration = std::make_unique<VehicleAcceleration>();
    // setup your test
    SECTION("Test property Longitudinal") {
        // Do implement test here
        testVehicleAcceleration->setLongitudinal(0.0f);
        REQUIRE( testVehicleAcceleration->getLongitudinal() == Approx( 0.0f ) );
    }
    SECTION("Test property Lateral") {
        // Do implement test here
        testVehicleAcceleration->setLateral(0.0f);
        REQUIRE( testVehicleAcceleration->getLateral() == Approx( 0.0f ) );
    }
    SECTION("Test property Vertical") {
        // Do implement test here
        testVehicleAcceleration->setVertical(0.0f);
        REQUIRE( testVehicleAcceleration->getVertical() == Approx( 0.0f ) );
    }
}
