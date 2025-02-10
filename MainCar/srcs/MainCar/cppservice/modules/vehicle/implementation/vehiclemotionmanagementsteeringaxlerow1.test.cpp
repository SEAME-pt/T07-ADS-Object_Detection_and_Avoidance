#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementsteeringaxlerow1.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementSteeringAxleRow1", "[VehicleMotionManagementSteeringAxleRow1]"){
    std::unique_ptr<IVehicleMotionManagementSteeringAxleRow1> testVehicleMotionManagementSteeringAxleRow1 = std::make_unique<VehicleMotionManagementSteeringAxleRow1>();
    // setup your test
    SECTION("Test property RackPositionOffsetTarget") {
        // Do implement test here
        testVehicleMotionManagementSteeringAxleRow1->setRackPositionOffsetTarget(0);
        REQUIRE( testVehicleMotionManagementSteeringAxleRow1->getRackPositionOffsetTarget() == 0 );
    }
    SECTION("Test property RackPositionOffsetTargetMode") {
        // Do implement test here
        testVehicleMotionManagementSteeringAxleRow1->setRackPositionOffsetTargetMode(0);
        REQUIRE( testVehicleMotionManagementSteeringAxleRow1->getRackPositionOffsetTargetMode() == 0 );
    }
    SECTION("Test property RackPosition") {
        // Do implement test here
        testVehicleMotionManagementSteeringAxleRow1->setRackPosition(0);
        REQUIRE( testVehicleMotionManagementSteeringAxleRow1->getRackPosition() == 0 );
    }
    SECTION("Test property RackPositionTarget") {
        // Do implement test here
        testVehicleMotionManagementSteeringAxleRow1->setRackPositionTarget(0);
        REQUIRE( testVehicleMotionManagementSteeringAxleRow1->getRackPositionTarget() == 0 );
    }
    SECTION("Test property RackPositionTargetMode") {
        // Do implement test here
        testVehicleMotionManagementSteeringAxleRow1->setRackPositionTargetMode(0);
        REQUIRE( testVehicleMotionManagementSteeringAxleRow1->getRackPositionTargetMode() == 0 );
    }
}
