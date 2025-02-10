#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2wheelleft.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementBrakeAxleRow2WheelLeft", "[VehicleMotionManagementBrakeAxleRow2WheelLeft]"){
    std::unique_ptr<IVehicleMotionManagementBrakeAxleRow2WheelLeft> testVehicleMotionManagementBrakeAxleRow2WheelLeft = std::make_unique<VehicleMotionManagementBrakeAxleRow2WheelLeft>();
    // setup your test
    SECTION("Test property Torque") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelLeft->setTorque(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelLeft->getTorque() == 0 );
    }
    SECTION("Test property TorqueArbitrated") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelLeft->setTorqueArbitrated(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelLeft->getTorqueArbitrated() == 0 );
    }
    SECTION("Test property TorqueFrictionMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelLeft->setTorqueFrictionMaximum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelLeft->getTorqueFrictionMaximum() == 0 );
    }
    SECTION("Test property TorqueFrictionMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelLeft->setTorqueFrictionMinimum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelLeft->getTorqueFrictionMinimum() == 0 );
    }
    SECTION("Test property OmegaUpper") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelLeft->setOmegaUpper(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelLeft->getOmegaUpper() == 0 );
    }
    SECTION("Test property OmegaLower") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelLeft->setOmegaLower(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelLeft->getOmegaLower() == 0 );
    }
}
