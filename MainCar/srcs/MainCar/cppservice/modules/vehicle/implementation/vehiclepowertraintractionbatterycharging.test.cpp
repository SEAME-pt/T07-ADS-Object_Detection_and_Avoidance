#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbatterycharging.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBatteryCharging", "[VehiclePowertrainTractionBatteryCharging]"){
    std::unique_ptr<IVehiclePowertrainTractionBatteryCharging> testVehiclePowertrainTractionBatteryCharging = std::make_unique<VehiclePowertrainTractionBatteryCharging>();
    // setup your test
    SECTION("Test property ChargeLimit") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setChargeLimit(0);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getChargeLimit() == 0 );
    }
    SECTION("Test property IsCharging") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setIsCharging(false);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getIsCharging() == false );
    }
    SECTION("Test property IsDischarging") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setIsDischarging(false);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getIsDischarging() == false );
    }
    SECTION("Test property StartStopCharging") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setStartStopCharging(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum::START);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getStartStopCharging() == Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum::START );
    }
    SECTION("Test property AveragePower") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setAveragePower(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getAveragePower() == Approx( 0.0f ) );
    }
    SECTION("Test property MaxPower") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setMaxPower(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getMaxPower() == Approx( 0.0f ) );
    }
    SECTION("Test property EvseId") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setEvseId(std::string());
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getEvseId() == std::string() );
    }
    SECTION("Test property PowerLoss") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setPowerLoss(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getPowerLoss() == Approx( 0.0f ) );
    }
    SECTION("Test property Temperature") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setTemperature(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getTemperature() == Approx( 0.0f ) );
    }
    SECTION("Test property ChargeRate") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setChargeRate(0.0f);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getChargeRate() == Approx( 0.0f ) );
    }
    SECTION("Test property TimeToComplete") {
        // Do implement test here
        testVehiclePowertrainTractionBatteryCharging->setTimeToComplete(0);
        REQUIRE( testVehiclePowertrainTractionBatteryCharging->getTimeToComplete() == 0 );
    }
}
