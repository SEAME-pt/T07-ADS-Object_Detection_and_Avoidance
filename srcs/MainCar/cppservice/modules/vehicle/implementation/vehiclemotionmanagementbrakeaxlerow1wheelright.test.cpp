#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1wheelright.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementBrakeAxleRow1WheelRight", "[VehicleMotionManagementBrakeAxleRow1WheelRight]"){
    std::unique_ptr<IVehicleMotionManagementBrakeAxleRow1WheelRight> testVehicleMotionManagementBrakeAxleRow1WheelRight = std::make_unique<VehicleMotionManagementBrakeAxleRow1WheelRight>();
    // setup your test
    SECTION("Test property Torque") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelRight->setTorque(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelRight->getTorque() == 0 );
    }
    SECTION("Test property TorqueArbitrated") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelRight->setTorqueArbitrated(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelRight->getTorqueArbitrated() == 0 );
    }
    SECTION("Test property TorqueFrictionMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelRight->setTorqueFrictionMaximum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelRight->getTorqueFrictionMaximum() == 0 );
    }
    SECTION("Test property TorqueFrictionMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelRight->setTorqueFrictionMinimum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelRight->getTorqueFrictionMinimum() == 0 );
    }
    SECTION("Test property OmegaUpper") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelRight->setOmegaUpper(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelRight->getOmegaUpper() == 0 );
    }
    SECTION("Test property OmegaLower") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1WheelRight->setOmegaLower(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1WheelRight->getOmegaLower() == 0 );
    }
}
