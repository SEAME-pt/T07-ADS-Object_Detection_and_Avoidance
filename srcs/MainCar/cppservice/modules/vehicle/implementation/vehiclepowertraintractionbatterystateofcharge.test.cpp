#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterystateofcharge.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryStateOfCharge", "[VehiclePowertrainTractionBatteryStateOfCharge]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryStateOfCharge> testVehiclePowertrainTractionBatteryStateOfCharge = std::make_unique<VehiclePowertrainTractionBatteryStateOfCharge>();
    // setup your test
    SECTION("Test property Current") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryStateOfCharge->setCurrent(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryStateOfCharge->getCurrent() == Approx( 0.0f ) );
    }
    SECTION("Test property Displayed") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryStateOfCharge->setDisplayed(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryStateOfCharge->getDisplayed() == Approx( 0.0f ) );
    }
    SECTION("Test property CurrentEnergy") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryStateOfCharge->setCurrentEnergy(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryStateOfCharge->getCurrentEnergy() == Approx( 0.0f ) );
    }
}
