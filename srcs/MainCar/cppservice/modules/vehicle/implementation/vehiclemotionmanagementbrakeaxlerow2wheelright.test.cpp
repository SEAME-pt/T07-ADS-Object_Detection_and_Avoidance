#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2wheelright.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementBrakeAxleRow2WheelRight", "[VehicleMotionManagementBrakeAxleRow2WheelRight]"){
    std::unique_ptr<IVehicleMotionManagementBrakeAxleRow2WheelRight> testVehicleMotionManagementBrakeAxleRow2WheelRight = std::make_unique<VehicleMotionManagementBrakeAxleRow2WheelRight>();
    // setup your test
    SECTION("Test property Torque") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelRight->setTorque(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelRight->getTorque() == 0 );
    }
    SECTION("Test property TorqueArbitrated") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelRight->setTorqueArbitrated(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelRight->getTorqueArbitrated() == 0 );
    }
    SECTION("Test property TorqueFrictionMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelRight->setTorqueFrictionMaximum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelRight->getTorqueFrictionMaximum() == 0 );
    }
    SECTION("Test property TorqueFrictionMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelRight->setTorqueFrictionMinimum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelRight->getTorqueFrictionMinimum() == 0 );
    }
    SECTION("Test property OmegaUpper") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelRight->setOmegaUpper(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelRight->getOmegaUpper() == 0 );
    }
    SECTION("Test property OmegaLower") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2WheelRight->setOmegaLower(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2WheelRight->getOmegaLower() == 0 );
    }
}
