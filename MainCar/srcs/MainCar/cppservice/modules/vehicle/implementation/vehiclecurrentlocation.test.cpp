#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclecurrentlocation.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleCurrentLocation", "[VehicleCurrentLocation]"){
    std::unique_ptr<IVehicleCurrentLocation> testVehicleCurrentLocation = std::make_unique<VehicleCurrentLocation>();
    // setup your test
    SECTION("Test property Timestamp") {
        // Do implement test here
        testVehicleCurrentLocation->setTimestamp(std::string());
        REQUIRE( testVehicleCurrentLocation->getTimestamp() == std::string() );
    }
    SECTION("Test property Latitude") {
        // Do implement test here
        testVehicleCurrentLocation->setLatitude(0.0);
        REQUIRE( testVehicleCurrentLocation->getLatitude() == 0.0 );
    }
    SECTION("Test property Longitude") {
        // Do implement test here
        testVehicleCurrentLocation->setLongitude(0.0);
        REQUIRE( testVehicleCurrentLocation->getLongitude() == 0.0 );
    }
    SECTION("Test property Heading") {
        // Do implement test here
        testVehicleCurrentLocation->setHeading(0.0);
        REQUIRE( testVehicleCurrentLocation->getHeading() == 0.0 );
    }
    SECTION("Test property HorizontalAccuracy") {
        // Do implement test here
        testVehicleCurrentLocation->setHorizontalAccuracy(0.0);
        REQUIRE( testVehicleCurrentLocation->getHorizontalAccuracy() == 0.0 );
    }
    SECTION("Test property Altitude") {
        // Do implement test here
        testVehicleCurrentLocation->setAltitude(0.0);
        REQUIRE( testVehicleCurrentLocation->getAltitude() == 0.0 );
    }
    SECTION("Test property VerticalAccuracy") {
        // Do implement test here
        testVehicleCurrentLocation->setVerticalAccuracy(0.0);
        REQUIRE( testVehicleCurrentLocation->getVerticalAccuracy() == 0.0 );
    }
}
