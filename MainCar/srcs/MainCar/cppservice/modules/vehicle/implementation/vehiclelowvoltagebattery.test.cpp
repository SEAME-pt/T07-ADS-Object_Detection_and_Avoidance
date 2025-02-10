#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclelowvoltagebattery.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleLowVoltageBattery", "[VehicleLowVoltageBattery]"){
    std::unique_ptr<IVehicleLowVoltageBattery> testVehicleLowVoltageBattery = std::make_unique<VehicleLowVoltageBattery>();
    // setup your test
    SECTION("Test property NominalVoltage") {
        // Do implement test here
        testVehicleLowVoltageBattery->setNominalVoltage(0);
        REQUIRE( testVehicleLowVoltageBattery->getNominalVoltage() == 0 );
    }
    SECTION("Test property NominalCapacity") {
        // Do implement test here
        testVehicleLowVoltageBattery->setNominalCapacity(0);
        REQUIRE( testVehicleLowVoltageBattery->getNominalCapacity() == 0 );
    }
    SECTION("Test property CurrentVoltage") {
        // Do implement test here
        testVehicleLowVoltageBattery->setCurrentVoltage(0.0f);
        REQUIRE( testVehicleLowVoltageBattery->getCurrentVoltage() == Approx( 0.0f ) );
    }
    SECTION("Test property CurrentCurrent") {
        // Do implement test here
        testVehicleLowVoltageBattery->setCurrentCurrent(0.0f);
        REQUIRE( testVehicleLowVoltageBattery->getCurrentCurrent() == Approx( 0.0f ) );
    }
}
