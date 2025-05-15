#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclecurrentlocationgnssreceivermountingposition.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleCurrentLocationGnssReceiverMountingPosition", "[VehicleCurrentLocationGnssReceiverMountingPosition]"){
    std::unique_ptr<IVehicleCurrentLocationGnssReceiverMountingPosition> testVehicleCurrentLocationGnssReceiverMountingPosition = std::make_unique<VehicleCurrentLocationGnssReceiverMountingPosition>();
    // setup your test
    SECTION("Test property X") {
        // Do implement test here
        testVehicleCurrentLocationGnssReceiverMountingPosition->setX(0);
        REQUIRE( testVehicleCurrentLocationGnssReceiverMountingPosition->getX() == 0 );
    }
    SECTION("Test property Y") {
        // Do implement test here
        testVehicleCurrentLocationGnssReceiverMountingPosition->setY(0);
        REQUIRE( testVehicleCurrentLocationGnssReceiverMountingPosition->getY() == 0 );
    }
    SECTION("Test property Z") {
        // Do implement test here
        testVehicleCurrentLocationGnssReceiverMountingPosition->setZ(0);
        REQUIRE( testVehicleCurrentLocationGnssReceiverMountingPosition->getZ() == 0 );
    }
}
