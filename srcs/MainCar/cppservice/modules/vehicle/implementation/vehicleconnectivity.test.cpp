#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehicleconnectivity.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleConnectivity", "[VehicleConnectivity]"){
    std::unique_ptr<IVehicleConnectivity> testVehicleConnectivity = std::make_unique<VehicleConnectivity>();
    // setup your test
    SECTION("Test property IsConnectivityAvailable") {
        // Do implement test here
        testVehicleConnectivity->setIsConnectivityAvailable(false);
        REQUIRE( testVehicleConnectivity->getIsConnectivityAvailable() == false );
    }
}
