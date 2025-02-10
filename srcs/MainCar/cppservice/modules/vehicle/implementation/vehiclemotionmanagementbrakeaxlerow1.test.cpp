#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementBrakeAxleRow1", "[VehicleMotionManagementBrakeAxleRow1]"){
    std::unique_ptr<IVehicleMotionManagementBrakeAxleRow1> testVehicleMotionManagementBrakeAxleRow1 = std::make_unique<VehicleMotionManagementBrakeAxleRow1>();
    // setup your test
    SECTION("Test property TorqueElectricMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1->setTorqueElectricMinimum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1->getTorqueElectricMinimum() == 0 );
    }
    SECTION("Test property TorqueFrictionDifferenceMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1->setTorqueFrictionDifferenceMaximum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1->getTorqueFrictionDifferenceMaximum() == 0 );
    }
    SECTION("Test property TorqueDistributionFrictionRightMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1->setTorqueDistributionFrictionRightMaximum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1->getTorqueDistributionFrictionRightMaximum() == 0 );
    }
    SECTION("Test property TorqueDistributionFrictionRightMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrakeAxleRow1->setTorqueDistributionFrictionRightMinimum(0);
        REQUIRE( testVehicleMotionManagementBrakeAxleRow1->getTorqueDistributionFrictionRightMinimum() == 0 );
    }
}
