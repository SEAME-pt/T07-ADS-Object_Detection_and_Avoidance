#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterydcdc.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryDcdc", "[VehiclePowertrainTractionBatteryDcdc]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryDcdc> testVehiclePowertrainTractionBatteryDcdc = std::make_unique<VehiclePowertrainTractionBatteryDcdc>();
    // setup your test
    SECTION("Test property PowerLoss") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryDcdc->setPowerLoss(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryDcdc->getPowerLoss() == Approx( 0.0f ) );
    }
    SECTION("Test property Temperature") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryDcdc->setTemperature(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryDcdc->getTemperature() == Approx( 0.0f ) );
    }
}
