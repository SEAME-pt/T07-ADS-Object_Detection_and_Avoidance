#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclediagnostics.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleDiagnostics", "[VehicleDiagnostics]"){
    std::unique_ptr<IVehicleDiagnostics> testVehicleDiagnostics = std::make_unique<VehicleDiagnostics>();
    // setup your test
    SECTION("Test property DTCCount") {
        // Do implement test here
        testVehicleDiagnostics->setDtcCount(0);
        REQUIRE( testVehicleDiagnostics->getDtcCount() == 0 );
    }
    SECTION("Test property DTCList") {
        // Do implement test here
        testVehicleDiagnostics->setDtcList(std::list<std::string>());
        REQUIRE( testVehicleDiagnostics->getDtcList() == std::list<std::string>() );
    }
}
