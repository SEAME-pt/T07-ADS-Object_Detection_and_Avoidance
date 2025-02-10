#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclemotionmanagementbrake.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleMotionManagementBrake", "[VehicleMotionManagementBrake]"){
    std::unique_ptr<IVehicleMotionManagementBrake> testVehicleMotionManagementBrake = std::make_unique<VehicleMotionManagementBrake>();
    // setup your test
    SECTION("Test property VehicleForceMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrake->setVehicleForceMaximum(0);
        REQUIRE( testVehicleMotionManagementBrake->getVehicleForceMaximum() == 0 );
    }
    SECTION("Test property VehicleForceElectricMinimumArbitrated") {
        // Do implement test here
        testVehicleMotionManagementBrake->setVehicleForceElectricMinimumArbitrated(0);
        REQUIRE( testVehicleMotionManagementBrake->getVehicleForceElectricMinimumArbitrated() == 0 );
    }
    SECTION("Test property VehicleForceElectric") {
        // Do implement test here
        testVehicleMotionManagementBrake->setVehicleForceElectric(0);
        REQUIRE( testVehicleMotionManagementBrake->getVehicleForceElectric() == 0 );
    }
    SECTION("Test property VehicleForceDistributionFrontMaximum") {
        // Do implement test here
        testVehicleMotionManagementBrake->setVehicleForceDistributionFrontMaximum(0);
        REQUIRE( testVehicleMotionManagementBrake->getVehicleForceDistributionFrontMaximum() == 0 );
    }
    SECTION("Test property VehicleForceDistributionFrontMinimum") {
        // Do implement test here
        testVehicleMotionManagementBrake->setVehicleForceDistributionFrontMinimum(0);
        REQUIRE( testVehicleMotionManagementBrake->getVehicleForceDistributionFrontMinimum() == 0 );
    }
}
