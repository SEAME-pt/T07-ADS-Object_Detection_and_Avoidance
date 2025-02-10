#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclebodyhorn.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleBodyHorn", "[VehicleBodyHorn]"){
    std::unique_ptr<IVehicleBodyHorn> testVehicleBodyHorn = std::make_unique<VehicleBodyHorn>();
    // setup your test
    SECTION("Test property IsActive") {
        // Do implement test here
        testVehicleBodyHorn->setIsActive(false);
        REQUIRE( testVehicleBodyHorn->getIsActive() == false );
    }
}
