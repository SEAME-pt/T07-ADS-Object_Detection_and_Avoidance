#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementBrakeAxleRow2", "[VehicleMotionManagementBrakeAxleRow2]"){
    std::unique_ptr<IVehicleMotionManagementBrakeAxleRow2> testVehicleMotionManagementBrakeAxleRow2 = std::make_unique<VehicleMotionManagementBrakeAxleRow2>();
    // setup your test
    SECTION("Test property TorqueElectricMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2->setTorqueElectricMinimum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2->getTorqueElectricMinimum() == 0 );
    }
    SECTION("Test property TorqueFrictionDifferenceMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2->setTorqueFrictionDifferenceMaximum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2->getTorqueFrictionDifferenceMaximum() == 0 );
    }
    SECTION("Test property TorqueDistributionFrictionRightMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2->setTorqueDistributionFrictionRightMaximum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2->getTorqueDistributionFrictionRightMaximum() == 0 );
    }
    SECTION("Test property TorqueDistributionFrictionRightMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow2->setTorqueDistributionFrictionRightMinimum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow2->getTorqueDistributionFrictionRightMinimum() == 0 );
    }
}
