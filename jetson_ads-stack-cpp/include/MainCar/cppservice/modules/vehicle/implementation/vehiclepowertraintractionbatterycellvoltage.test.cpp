#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterycellvoltage.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryCellVoltage", "[VehiclePowertrainTractionBatteryCellVoltage]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryCellVoltage> testVehiclePowertrainTractionBatteryCellVoltage = std::make_unique<VehiclePowertrainTractionBatteryCellVoltage>();
    // setup your test
    SECTION("Test property Min") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCellVoltage->setMin(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryCellVoltage->getMin() == Approx( 0.0f ) );
    }
    SECTION("Test property IdMin") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCellVoltage->setIdMin(0);
        REQUIRE( testVehiclePowertrainTractionBatteryCellVoltage->getIdMin() == 0 );
    }
    SECTION("Test property Max") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCellVoltage->setMax(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryCellVoltage->getMax() == Approx( 0.0f ) );
    }
    SECTION("Test property IdMax") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCellVoltage->setIdMax(0);
        REQUIRE( testVehiclePowertrainTractionBatteryCellVoltage->getIdMax() == 0 );
    }
    SECTION("Test property CellVoltages") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCellVoltage->setCellVoltages(std::list<float>());
        REQUIRE( testVehiclePowertrainTractionBatteryCellVoltage->getCellVoltages() == std::list<float>() );
    }
}
