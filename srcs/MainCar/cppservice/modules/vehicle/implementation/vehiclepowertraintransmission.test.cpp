#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertraintransmission.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainTransmission", "[VehiclePowertrainTransmission]"){
    std::unique_ptr<IVehiclePowertrainTransmission> testVehiclePowertrainTransmission = std::make_unique<VehiclePowertrainTransmission>();
    // setup your test
    SECTION("Test property Type") {
        // Do implement test here
        testVehiclePowertrainTransmission->setType(Vehicle_Powertrain_Transmission_Type_ValueEnum::UNKNOWN);
        REQUIRE( testVehiclePowertrainTransmission->getType() == Vehicle_Powertrain_Transmission_Type_ValueEnum::UNKNOWN );
    }
    SECTION("Test property GearCount") {
        // Do implement test here
        testVehiclePowertrainTransmission->setGearCount(0);
        REQUIRE( testVehiclePowertrainTransmission->getGearCount() == 0 );
    }
    SECTION("Test property DriveType") {
        // Do implement test here
        testVehiclePowertrainTransmission->setDriveType(Vehicle_Powertrain_Transmission_DriveType_ValueEnum::UNKNOWN);
        REQUIRE( testVehiclePowertrainTransmission->getDriveType() == Vehicle_Powertrain_Transmission_DriveType_ValueEnum::UNKNOWN );
    }
    SECTION("Test property TravelledDistance") {
        // Do implement test here
        testVehiclePowertrainTransmission->setTravelledDistance(0.0f);
        REQUIRE( testVehiclePowertrainTransmission->getTravelledDistance() == Approx( 0.0f ) );
    }
    SECTION("Test property CurrentGear") {
        // Do implement test here
        testVehiclePowertrainTransmission->setCurrentGear(0);
        REQUIRE( testVehiclePowertrainTransmission->getCurrentGear() == 0 );
    }
    SECTION("Test property SelectedGear") {
        // Do implement test here
        testVehiclePowertrainTransmission->setSelectedGear(0);
        REQUIRE( testVehiclePowertrainTransmission->getSelectedGear() == 0 );
    }
    SECTION("Test property IsParkLockEngaged") {
        // Do implement test here
        testVehiclePowertrainTransmission->setIsParkLockEngaged(false);
        REQUIRE( testVehiclePowertrainTransmission->getIsParkLockEngaged() == false );
    }
    SECTION("Test property IsLowRangeEngaged") {
        // Do implement test here
        testVehiclePowertrainTransmission->setIsLowRangeEngaged(false);
        REQUIRE( testVehiclePowertrainTransmission->getIsLowRangeEngaged() == false );
    }
    SECTION("Test property IsElectricalPowertrainEngaged") {
        // Do implement test here
        testVehiclePowertrainTransmission->setIsElectricalPowertrainEngaged(false);
        REQUIRE( testVehiclePowertrainTransmission->getIsElectricalPowertrainEngaged() == false );
    }
    SECTION("Test property PerformanceMode") {
        // Do implement test here
        testVehiclePowertrainTransmission->setPerformanceMode(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::NORMAL);
        REQUIRE( testVehiclePowertrainTransmission->getPerformanceMode() == Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::NORMAL );
    }
    SECTION("Test property GearChangeMode") {
        // Do implement test here
        testVehiclePowertrainTransmission->setGearChangeMode(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum::MANUAL);
        REQUIRE( testVehiclePowertrainTransmission->getGearChangeMode() == Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum::MANUAL );
    }
    SECTION("Test property Temperature") {
        // Do implement test here
        testVehiclePowertrainTransmission->setTemperature(0.0f);
        REQUIRE( testVehiclePowertrainTransmission->getTemperature() == Approx( 0.0f ) );
    }
    SECTION("Test property ClutchEngagement") {
        // Do implement test here
        testVehiclePowertrainTransmission->setClutchEngagement(0.0f);
        REQUIRE( testVehiclePowertrainTransmission->getClutchEngagement() == Approx( 0.0f ) );
    }
    SECTION("Test property ClutchWear") {
        // Do implement test here
        testVehiclePowertrainTransmission->setClutchWear(0);
        REQUIRE( testVehiclePowertrainTransmission->getClutchWear() == 0 );
    }
    SECTION("Test property DiffLockFrontEngagement") {
        // Do implement test here
        testVehiclePowertrainTransmission->setDiffLockFrontEngagement(0.0f);
        REQUIRE( testVehiclePowertrainTransmission->getDiffLockFrontEngagement() == Approx( 0.0f ) );
    }
    SECTION("Test property DiffLockRearEngagement") {
        // Do implement test here
        testVehiclePowertrainTransmission->setDiffLockRearEngagement(0.0f);
        REQUIRE( testVehiclePowertrainTransmission->getDiffLockRearEngagement() == Approx( 0.0f ) );
    }
    SECTION("Test property TorqueDistribution") {
        // Do implement test here
        testVehiclePowertrainTransmission->setTorqueDistribution(0.0f);
        REQUIRE( testVehiclePowertrainTransmission->getTorqueDistribution() == Approx( 0.0f ) );
    }
}
