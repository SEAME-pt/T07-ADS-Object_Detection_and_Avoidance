#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterycharginglocation.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryChargingLocation", "[VehiclePowertrainTractionBatteryChargingLocation]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingLocation> testVehiclePowertrainTractionBatteryChargingLocation = std::make_unique<VehiclePowertrainTractionBatteryChargingLocation>();
    // setup your test
    SECTION("Test property Latitude") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingLocation->setLatitude(0.0);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingLocation->getLatitude() == 0.0 );
    }
    SECTION("Test property Longitude") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingLocation->setLongitude(0.0);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingLocation->getLongitude() == 0.0 );
    }
    SECTION("Test property Altitude") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryChargingLocation->setAltitude(0.0);
        REQUIRE( testVehiclePowertrainTractionBatteryChargingLocation->getAltitude() == 0.0 );
    }
}
