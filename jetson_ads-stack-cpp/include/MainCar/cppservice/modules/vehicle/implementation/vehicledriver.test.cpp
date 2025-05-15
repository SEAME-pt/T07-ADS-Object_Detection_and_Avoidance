#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehicledriver.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleDriver", "[VehicleDriver]"){
    std::unique_ptr<IVehicleDriver> testVehicleDriver = std::make_unique<VehicleDriver>();
    // setup your test
    SECTION("Test property DistractionLevel") {
        // Do implement test here
        testVehicleDriver->setDistractionLevel(0.0f);
        REQUIRE( testVehicleDriver->getDistractionLevel() == Approx( 0.0f ) );
    }
    SECTION("Test property IsEyesOnRoad") {
        // Do implement test here
        testVehicleDriver->setIsEyesOnRoad(false);
        REQUIRE( testVehicleDriver->getIsEyesOnRoad() == false );
    }
    SECTION("Test property IsHandsOnWheel") {
        // Do implement test here
        testVehicleDriver->setIsHandsOnWheel(false);
        REQUIRE( testVehicleDriver->getIsHandsOnWheel() == false );
    }
    SECTION("Test property AttentiveProbability") {
        // Do implement test here
        testVehicleDriver->setAttentiveProbability(0.0f);
        REQUIRE( testVehicleDriver->getAttentiveProbability() == Approx( 0.0f ) );
    }
    SECTION("Test property FatigueLevel") {
        // Do implement test here
        testVehicleDriver->setFatigueLevel(0.0f);
        REQUIRE( testVehicleDriver->getFatigueLevel() == Approx( 0.0f ) );
    }
    SECTION("Test property HeartRate") {
        // Do implement test here
        testVehicleDriver->setHeartRate(0);
        REQUIRE( testVehicleDriver->getHeartRate() == 0 );
    }
}
