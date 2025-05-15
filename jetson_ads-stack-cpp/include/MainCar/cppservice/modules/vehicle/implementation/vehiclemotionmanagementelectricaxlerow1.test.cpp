#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementelectricaxlerow1.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementElectricAxleRow1", "[VehicleMotionManagementElectricAxleRow1]"){
    std::unique_ptr<IVehicleMotionManagementElectricAxleRow1> testVehicleMotionManagementElectricAxleRow1 = std::make_unique<VehicleMotionManagementElectricAxleRow1>();
    // setup your test
    SECTION("Test property Torque") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorque(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorque() == 0 );
    }
    SECTION("Test property TorqueTarget") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueTarget(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueTarget() == 0 );
    }
    SECTION("Test property TorqueMaximum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueMaximum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueMaximum() == 0 );
    }
    SECTION("Test property TorqueMinimum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueMinimum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueMinimum() == 0 );
    }
    SECTION("Test property TorqueLongTermMaximum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueLongTermMaximum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueLongTermMaximum() == 0 );
    }
    SECTION("Test property TorqueLongTermMinimum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueLongTermMinimum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueLongTermMinimum() == 0 );
    }
    SECTION("Test property TorqueShortTermMaximum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueShortTermMaximum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueShortTermMaximum() == 0 );
    }
    SECTION("Test property TorqueShortTermMinimum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueShortTermMinimum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueShortTermMinimum() == 0 );
    }
    SECTION("Test property TorqueMaximumLimit") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueMaximumLimit(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueMaximumLimit() == 0 );
    }
    SECTION("Test property TorqueMinimumLimit") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setTorqueMinimumLimit(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getTorqueMinimumLimit() == 0 );
    }
    SECTION("Test property RotationalSpeed") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setRotationalSpeed(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getRotationalSpeed() == 0 );
    }
    SECTION("Test property RotationalSpeedTarget") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setRotationalSpeedTarget(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getRotationalSpeedTarget() == 0 );
    }
    SECTION("Test property RotationalSpeedMaximumLimit") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setRotationalSpeedMaximumLimit(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getRotationalSpeedMaximumLimit() == 0 );
    }
    SECTION("Test property RotationalSpeedMinimumLimit") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow1->setRotationalSpeedMinimumLimit(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow1->getRotationalSpeedMinimumLimit() == 0 );
    }
}
