#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertrain.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrain", "[VehiclePowertrain]"){
    std::unique_ptr<IVehiclePowertrain> testVehiclePowertrain = std::make_unique<VehiclePowertrain>();
    // setup your test
    SECTION("Test property AccumulatedBrakingEnergy") {
        // Do implement test here
        testVehiclePowertrain->setAccumulatedBrakingEnergy(0.0f);
        REQUIRE( testVehiclePowertrain->getAccumulatedBrakingEnergy() == Approx( 0.0f ) );
    }
    SECTION("Test property Range") {
        // Do implement test here
        testVehiclePowertrain->setRange(0);
        REQUIRE( testVehiclePowertrain->getRange() == 0 );
    }
    SECTION("Test property TimeRemaining") {
        // Do implement test here
        testVehiclePowertrain->setTimeRemaining(0);
        REQUIRE( testVehiclePowertrain->getTimeRemaining() == 0 );
    }
    SECTION("Test property Type") {
        // Do implement test here
        testVehiclePowertrain->setType(Vehicle_Powertrain_Type_ValueEnum::COMBUSTION);
        REQUIRE( testVehiclePowertrain->getType() == Vehicle_Powertrain_Type_ValueEnum::COMBUSTION );
    }
    SECTION("Test property PowerOptimizeLevel") {
        // Do implement test here
        testVehiclePowertrain->setPowerOptimizeLevel(0);
        REQUIRE( testVehiclePowertrain->getPowerOptimizeLevel() == 0 );
    }
    SECTION("Test property IsAutoPowerOptimize") {
        // Do implement test here
        testVehiclePowertrain->setIsAutoPowerOptimize(false);
        REQUIRE( testVehiclePowertrain->getIsAutoPowerOptimize() == false );
    }
}
