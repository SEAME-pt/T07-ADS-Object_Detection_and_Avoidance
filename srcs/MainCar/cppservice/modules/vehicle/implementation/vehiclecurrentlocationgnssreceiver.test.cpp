#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclecurrentlocationgnssreceiver.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleCurrentLocationGnssReceiver", "[VehicleCurrentLocationGnssReceiver]"){
    std::unique_ptr<IVehicleCurrentLocationGnssReceiver> testVehicleCurrentLocationGnssReceiver = std::make_unique<VehicleCurrentLocationGnssReceiver>();
    // setup your test
    SECTION("Test property FixType") {
        // Do implement test here
        testVehicleCurrentLocationGnssReceiver->setFixType(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::NONE);
        REQUIRE( testVehicleCurrentLocationGnssReceiver->getFixType() == Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::NONE );
    }
}
