#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclepowertrainelectricmotorenginecoolant.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehiclePowertrainElectricMotorEngineCoolant", "[VehiclePowertrainElectricMotorEngineCoolant]"){
    std::unique_ptr<IVehiclePowertrainElectricMotorEngineCoolant> testVehiclePowertrainElectricMotorEngineCoolant = std::make_unique<VehiclePowertrainElectricMotorEngineCoolant>();
    // setup your test
    SECTION("Test property Capacity") {
        // Do implement test here
        testVehiclePowertrainElectricMotorEngineCoolant->setCapacity(0.0f);
        REQUIRE( testVehiclePowertrainElectricMotorEngineCoolant->getCapacity() == Approx( 0.0f ) );
    }
    SECTION("Test property Temperature") {
        // Do implement test here
        testVehiclePowertrainElectricMotorEngineCoolant->setTemperature(0.0f);
        REQUIRE( testVehiclePowertrainElectricMotorEngineCoolant->getTemperature() == Approx( 0.0f ) );
    }
    SECTION("Test property Level") {
        // Do implement test here
        testVehiclePowertrainElectricMotorEngineCoolant->setLevel(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum::CRITICALLY_LOW);
        REQUIRE( testVehiclePowertrainElectricMotorEngineCoolant->getLevel() == Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum::CRITICALLY_LOW );
    }
    SECTION("Test property LifeRemaining") {
        // Do implement test here
        testVehiclePowertrainElectricMotorEngineCoolant->setLifeRemaining(0);
        REQUIRE( testVehiclePowertrainElectricMotorEngineCoolant->getLifeRemaining() == 0 );
    }
}
