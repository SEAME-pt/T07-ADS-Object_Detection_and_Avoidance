#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehicle.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing Vehicle", "[Vehicle]"){
    std::unique_ptr<IVehicle> testVehicle = std::make_unique<Vehicle>();
    // setup your test
    SECTION("Test property LowVoltageSystemState") {
        // Do implement test here
        testVehicle->setLowVoltageSystemState(Vehicle_LowVoltageSystemState_ValueEnum::UNDEFINED);
        REQUIRE( testVehicle->getLowVoltageSystemState() == Vehicle_LowVoltageSystemState_ValueEnum::UNDEFINED );
    }
    SECTION("Test property Speed") {
        // Do implement test here
        testVehicle->setSpeed(0.0f);
        REQUIRE( testVehicle->getSpeed() == Approx( 0.0f ) );
    }
    SECTION("Test property TraveledDistance") {
        // Do implement test here
        testVehicle->setTraveledDistance(0.0f);
        REQUIRE( testVehicle->getTraveledDistance() == Approx( 0.0f ) );
    }
    SECTION("Test property TraveledDistanceSinceStart") {
        // Do implement test here
        testVehicle->setTraveledDistanceSinceStart(0.0f);
        REQUIRE( testVehicle->getTraveledDistanceSinceStart() == Approx( 0.0f ) );
    }
    SECTION("Test property StartTime") {
        // Do implement test here
        testVehicle->setStartTime(std::string());
        REQUIRE( testVehicle->getStartTime() == std::string() );
    }
    SECTION("Test property TripDuration") {
        // Do implement test here
        testVehicle->setTripDuration(0.0f);
        REQUIRE( testVehicle->getTripDuration() == Approx( 0.0f ) );
    }
    SECTION("Test property TripMeterReading") {
        // Do implement test here
        testVehicle->setTripMeterReading(0.0f);
        REQUIRE( testVehicle->getTripMeterReading() == Approx( 0.0f ) );
    }
    SECTION("Test property IsBrokenDown") {
        // Do implement test here
        testVehicle->setIsBrokenDown(false);
        REQUIRE( testVehicle->getIsBrokenDown() == false );
    }
    SECTION("Test property IsMoving") {
        // Do implement test here
        testVehicle->setIsMoving(false);
        REQUIRE( testVehicle->getIsMoving() == false );
    }
    SECTION("Test property AverageSpeed") {
        // Do implement test here
        testVehicle->setAverageSpeed(0.0f);
        REQUIRE( testVehicle->getAverageSpeed() == Approx( 0.0f ) );
    }
    SECTION("Test property PowerOptimizeLevel") {
        // Do implement test here
        testVehicle->setPowerOptimizeLevel(0);
        REQUIRE( testVehicle->getPowerOptimizeLevel() == 0 );
    }
    SECTION("Test property IsAutoPowerOptimize") {
        // Do implement test here
        testVehicle->setIsAutoPowerOptimize(false);
        REQUIRE( testVehicle->getIsAutoPowerOptimize() == false );
    }
}
