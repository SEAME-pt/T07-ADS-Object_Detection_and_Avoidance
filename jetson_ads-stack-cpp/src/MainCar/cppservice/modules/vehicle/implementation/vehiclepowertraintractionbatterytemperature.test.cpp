#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterytemperature.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryTemperature", "[VehiclePowertrainTractionBatteryTemperature]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryTemperature> testVehiclePowertrainTractionBatteryTemperature = std::make_unique<VehiclePowertrainTractionBatteryTemperature>();
    // setup your test
    SECTION("Test property Average") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryTemperature->setAverage(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryTemperature->getAverage() == Approx( 0.0f ) );
    }
    SECTION("Test property Min") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryTemperature->setMin(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryTemperature->getMin() == Approx( 0.0f ) );
    }
    SECTION("Test property Max") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryTemperature->setMax(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryTemperature->getMax() == Approx( 0.0f ) );
    }
    SECTION("Test property CellTemperature") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryTemperature->setCellTemperature(std::list<float>());
        REQUIRE( testVehiclePowertrainTractionBatteryTemperature->getCellTemperature() == std::list<float>() );
    }
}
