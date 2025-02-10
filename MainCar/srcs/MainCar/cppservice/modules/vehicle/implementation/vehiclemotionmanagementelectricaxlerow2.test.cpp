#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementelectricaxlerow2.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementElectricAxleRow2", "[VehicleMotionManagementElectricAxleRow2]"){
    std::unique_ptr<IVehicleMotionManagementElectricAxleRow2> testVehicleMotionManagementElectricAxleRow2 = std::make_unique<VehicleMotionManagementElectricAxleRow2>();
    // setup your test
    SECTION("Test property Torque") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorque(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorque() == 0 );
    }
    SECTION("Test property TorqueTarget") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueTarget(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueTarget() == 0 );
    }
    SECTION("Test property TorqueMaximum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueMaximum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueMaximum() == 0 );
    }
    SECTION("Test property TorqueMinimum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueMinimum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueMinimum() == 0 );
    }
    SECTION("Test property TorqueLongTermMaximum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueLongTermMaximum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueLongTermMaximum() == 0 );
    }
    SECTION("Test property TorqueLongTermMinimum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueLongTermMinimum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueLongTermMinimum() == 0 );
    }
    SECTION("Test property TorqueShortTermMaximum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueShortTermMaximum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueShortTermMaximum() == 0 );
    }
    SECTION("Test property TorqueShortTermMinimum") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueShortTermMinimum(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueShortTermMinimum() == 0 );
    }
    SECTION("Test property TorqueMaximumLimit") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueMaximumLimit(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueMaximumLimit() == 0 );
    }
    SECTION("Test property TorqueMinimumLimit") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setTorqueMinimumLimit(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getTorqueMinimumLimit() == 0 );
    }
    SECTION("Test property RotationalSpeed") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setRotationalSpeed(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getRotationalSpeed() == 0 );
    }
    SECTION("Test property RotationalSpeedTarget") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setRotationalSpeedTarget(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getRotationalSpeedTarget() == 0 );
    }
    SECTION("Test property RotationalSpeedMaximumLimit") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setRotationalSpeedMaximumLimit(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getRotationalSpeedMaximumLimit() == 0 );
    }
    SECTION("Test property RotationalSpeedMinimumLimit") {
        // Do implement test here
        testVehicleMotionManagementElectricAxleRow2->setRotationalSpeedMinimumLimit(0);
        REQUIRE( testVehicleMotionManagementElectricAxleRow2->getRotationalSpeedMinimumLimit() == 0 );
    }
}
