#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehicleversionvss.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleVersionVss", "[VehicleVersionVss]"){
    std::unique_ptr<IVehicleVersionVss> testVehicleVersionVss = std::make_unique<VehicleVersionVss>();
    // setup your test
    SECTION("Test property Major") {
        // Do implement test here
        testVehicleVersionVss->setMajor(0);
        REQUIRE( testVehicleVersionVss->getMajor() == 0 );
    }
    SECTION("Test property Minor") {
        // Do implement test here
        testVehicleVersionVss->setMinor(0);
        REQUIRE( testVehicleVersionVss->getMinor() == 0 );
    }
    SECTION("Test property Patch") {
        // Do implement test here
        testVehicleVersionVss->setPatch(0);
        REQUIRE( testVehicleVersionVss->getPatch() == 0 );
    }
    SECTION("Test property Label") {
        // Do implement test here
        testVehicleVersionVss->setLabel(std::string());
        REQUIRE( testVehicleVersionVss->getLabel() == std::string() );
    }
}
