#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertrainelectricmotor.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainElectricMotor", "[VehiclePowertrainElectricMotor]"){
    std::unique_ptr<IVehiclePowertrainElectricMotor> testVehiclePowertrainElectricMotor = std::make_unique<VehiclePowertrainElectricMotor>();
    // setup your test
    SECTION("Test property EngineCode") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setEngineCode(std::string());
        REQUIRE( testVehiclePowertrainElectricMotor->getEngineCode() == std::string() );
    }
    SECTION("Test property MaxPower") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setMaxPower(0);
        REQUIRE( testVehiclePowertrainElectricMotor->getMaxPower() == 0 );
    }
    SECTION("Test property MaxTorque") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setMaxTorque(0);
        REQUIRE( testVehiclePowertrainElectricMotor->getMaxTorque() == 0 );
    }
    SECTION("Test property MaxRegenPower") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setMaxRegenPower(0);
        REQUIRE( testVehiclePowertrainElectricMotor->getMaxRegenPower() == 0 );
    }
    SECTION("Test property MaxRegenTorque") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setMaxRegenTorque(0);
        REQUIRE( testVehiclePowertrainElectricMotor->getMaxRegenTorque() == 0 );
    }
    SECTION("Test property Speed") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setSpeed(0);
        REQUIRE( testVehiclePowertrainElectricMotor->getSpeed() == 0 );
    }
    SECTION("Test property TimeInUse") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setTimeInUse(0.0f);
        REQUIRE( testVehiclePowertrainElectricMotor->getTimeInUse() == Approx( 0.0f ) );
    }
    SECTION("Test property Temperature") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setTemperature(0.0f);
        REQUIRE( testVehiclePowertrainElectricMotor->getTemperature() == Approx( 0.0f ) );
    }
    SECTION("Test property CoolantTemperature_Deprecated") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setCoolantTemperatureDeprecated(0.0f);
        REQUIRE( testVehiclePowertrainElectricMotor->getCoolantTemperatureDeprecated() == Approx( 0.0f ) );
    }
    SECTION("Test property Power") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setPower(0);
        REQUIRE( testVehiclePowertrainElectricMotor->getPower() == 0 );
    }
    SECTION("Test property Torque") {
        // Do implement test here
        testVehiclePowertrainElectricMotor->setTorque(0);
        REQUIRE( testVehiclePowertrainElectricMotor->getTorque() == 0 );
    }
}
