#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintractionbattery.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTractionBattery", "[VehiclePowertrainTractionBattery]"){
    std::unique_ptr<IVehiclePowertrainTractionBattery> testVehiclePowertrainTractionBattery = std::make_unique<VehiclePowertrainTractionBattery>();
    // setup your test
    SECTION("Test property Id") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setId(std::string());
        REQUIRE( testVehiclePowertrainTractionBattery->getId() == std::string() );
    }
    SECTION("Test property ProductionDate") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setProductionDate(std::string());
        REQUIRE( testVehiclePowertrainTractionBattery->getProductionDate() == std::string() );
    }
    SECTION("Test property ErrorCodes") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setErrorCodes(std::list<std::string>());
        REQUIRE( testVehiclePowertrainTractionBattery->getErrorCodes() == std::list<std::string>() );
    }
    SECTION("Test property IsPowerConnected") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setIsPowerConnected(false);
        REQUIRE( testVehiclePowertrainTractionBattery->getIsPowerConnected() == false );
    }
    SECTION("Test property IsGroundConnected") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setIsGroundConnected(false);
        REQUIRE( testVehiclePowertrainTractionBattery->getIsGroundConnected() == false );
    }
    SECTION("Test property GrossCapacity") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setGrossCapacity(0);
        REQUIRE( testVehiclePowertrainTractionBattery->getGrossCapacity() == 0 );
    }
    SECTION("Test property NetCapacity") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setNetCapacity(0);
        REQUIRE( testVehiclePowertrainTractionBattery->getNetCapacity() == 0 );
    }
    SECTION("Test property StateOfHealth") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setStateOfHealth(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getStateOfHealth() == Approx( 0.0f ) );
    }
    SECTION("Test property NominalVoltage") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setNominalVoltage(0);
        REQUIRE( testVehiclePowertrainTractionBattery->getNominalVoltage() == 0 );
    }
    SECTION("Test property MaxVoltage") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setMaxVoltage(0);
        REQUIRE( testVehiclePowertrainTractionBattery->getMaxVoltage() == 0 );
    }
    SECTION("Test property CurrentVoltage") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setCurrentVoltage(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getCurrentVoltage() == Approx( 0.0f ) );
    }
    SECTION("Test property CurrentCurrent") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setCurrentCurrent(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getCurrentCurrent() == Approx( 0.0f ) );
    }
    SECTION("Test property CurrentPower") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setCurrentPower(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getCurrentPower() == Approx( 0.0f ) );
    }
    SECTION("Test property AccumulatedChargedEnergy") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setAccumulatedChargedEnergy(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getAccumulatedChargedEnergy() == Approx( 0.0f ) );
    }
    SECTION("Test property AccumulatedConsumedEnergy") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setAccumulatedConsumedEnergy(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getAccumulatedConsumedEnergy() == Approx( 0.0f ) );
    }
    SECTION("Test property AccumulatedChargedThroughput") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setAccumulatedChargedThroughput(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getAccumulatedChargedThroughput() == Approx( 0.0f ) );
    }
    SECTION("Test property AccumulatedConsumedThroughput") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setAccumulatedConsumedThroughput(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getAccumulatedConsumedThroughput() == Approx( 0.0f ) );
    }
    SECTION("Test property PowerLoss") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setPowerLoss(0.0f);
        REQUIRE( testVehiclePowertrainTractionBattery->getPowerLoss() == Approx( 0.0f ) );
    }
    SECTION("Test property Range") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setRange(0);
        REQUIRE( testVehiclePowertrainTractionBattery->getRange() == 0 );
    }
    SECTION("Test property TimeRemaining") {
        // Do implement test here
        testVehiclePowertrainTractionBattery->setTimeRemaining(0);
        REQUIRE( testVehiclePowertrainTractionBattery->getTimeRemaining() == 0 );
    }
}
