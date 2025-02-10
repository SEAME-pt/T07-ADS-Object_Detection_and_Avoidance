#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementsteeringsteeringwheel.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementSteeringSteeringWheel", "[VehicleMotionManagementSteeringSteeringWheel]"){
    std::unique_ptr<IVehicleMotionManagementSteeringSteeringWheel> testVehicleMotionManagementSteeringSteeringWheel = std::make_unique<VehicleMotionManagementSteeringSteeringWheel>();
    // setup your test
    SECTION("Test property Angle") {
        // Do implement test here
        testVehicleMotionManagementSteeringSteeringWheel->setAngle(0);
        REQUIRE( testVehicleMotionManagementSteeringSteeringWheel->getAngle() == 0 );
    }
    SECTION("Test property AngleTarget") {
        // Do implement test here
        testVehicleMotionManagementSteeringSteeringWheel->setAngleTarget(0);
        REQUIRE( testVehicleMotionManagementSteeringSteeringWheel->getAngleTarget() == 0 );
    }
    SECTION("Test property AngleTargetMode") {
        // Do implement test here
        testVehicleMotionManagementSteeringSteeringWheel->setAngleTargetMode(0);
        REQUIRE( testVehicleMotionManagementSteeringSteeringWheel->getAngleTargetMode() == 0 );
    }
    SECTION("Test property TorqueOffsetTarget") {
        // Do implement test here
        testVehicleMotionManagementSteeringSteeringWheel->setTorqueOffsetTarget(0);
        REQUIRE( testVehicleMotionManagementSteeringSteeringWheel->getTorqueOffsetTarget() == 0 );
    }
    SECTION("Test property TorqueOffsetTargetMode") {
        // Do implement test here
        testVehicleMotionManagementSteeringSteeringWheel->setTorqueOffsetTargetMode(0);
        REQUIRE( testVehicleMotionManagementSteeringSteeringWheel->getTorqueOffsetTargetMode() == 0 );
    }
    SECTION("Test property TorqueTarget") {
        // Do implement test here
        testVehicleMotionManagementSteeringSteeringWheel->setTorqueTarget(0);
        REQUIRE( testVehicleMotionManagementSteeringSteeringWheel->getTorqueTarget() == 0 );
    }
    SECTION("Test property TorqueTargetMode") {
        // Do implement test here
        testVehicleMotionManagementSteeringSteeringWheel->setTorqueTargetMode(0);
        REQUIRE( testVehicleMotionManagementSteeringSteeringWheel->getTorqueTargetMode() == 0 );
    }
    SECTION("Test property Torque") {
        // Do implement test here
        testVehicleMotionManagementSteeringSteeringWheel->setTorque(0);
        REQUIRE( testVehicleMotionManagementSteeringSteeringWheel->getTorque() == 0 );
    }
}
