#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1wheelleft.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementBrakeAxleRow1WheelLeft", "[VehicleMotionManagementBrakeAxleRow1WheelLeft]"){
    std::unique_ptr<IVehicleMotionManagementBrakeAxleRow1WheelLeft> testVehicleMotionManagementBrakeAxleRow1WheelLeft = std::make_unique<VehicleMotionManagementBrakeAxleRow1WheelLeft>();
    // setup your test
    SECTION("Test property Torque") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelLeft->setTorque(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelLeft->getTorque() == 0 );
    }
    SECTION("Test property TorqueArbitrated") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelLeft->setTorqueArbitrated(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelLeft->getTorqueArbitrated() == 0 );
    }
    SECTION("Test property TorqueFrictionMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelLeft->setTorqueFrictionMaximum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelLeft->getTorqueFrictionMaximum() == 0 );
    }
    SECTION("Test property TorqueFrictionMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelLeft->setTorqueFrictionMinimum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelLeft->getTorqueFrictionMinimum() == 0 );
    }
    SECTION("Test property OmegaUpper") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelLeft->setOmegaUpper(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelLeft->getOmegaUpper() == 0 );
    }
    SECTION("Test property OmegaLower") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelLeft->setOmegaLower(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelLeft->getOmegaLower() == 0 );
    }
}
