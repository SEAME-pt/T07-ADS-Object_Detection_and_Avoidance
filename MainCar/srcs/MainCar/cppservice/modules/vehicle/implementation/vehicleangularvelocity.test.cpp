#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehicleangularvelocity.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleAngularVelocity", "[VehicleAngularVelocity]"){
    std::unique_ptr<IVehicleAngularVelocity> testVehicleAngularVelocity = std::make_unique<VehicleAngularVelocity>();
    // setup your test
    SECTION("Test property Roll") {
        // Do implement test here
        testVehicleAngularVelocity->setRoll(0.0f);
        REQUIRE( testVehicleAngularVelocity->getRoll() == Approx( 0.0f ) );
    }
    SECTION("Test property Pitch") {
        // Do implement test here
        testVehicleAngularVelocity->setPitch(0.0f);
        REQUIRE( testVehicleAngularVelocity->getPitch() == Approx( 0.0f ) );
    }
    SECTION("Test property Yaw") {
        // Do implement test here
        testVehicleAngularVelocity->setYaw(0.0f);
        REQUIRE( testVehicleAngularVelocity->getYaw() == Approx( 0.0f ) );
    }
}
