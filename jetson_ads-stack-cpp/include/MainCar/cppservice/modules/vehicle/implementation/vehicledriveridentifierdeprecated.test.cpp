#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehicledriveridentifierdeprecated.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleDriverIdentifierDeprecated", "[VehicleDriverIdentifierDeprecated]"){
    std::unique_ptr<IVehicleDriverIdentifierDeprecated> testVehicleDriverIdentifierDeprecated = std::make_unique<VehicleDriverIdentifierDeprecated>();
    // setup your test
    SECTION("Test property Subject_Deprecated") {
        // Do implement test here
        testVehicleDriverIdentifierDeprecated->setSubjectDeprecated(std::string());
        REQUIRE( testVehicleDriverIdentifierDeprecated->getSubjectDeprecated() == std::string() );
    }
    SECTION("Test property Issuer_Deprecated") {
        // Do implement test here
        testVehicleDriverIdentifierDeprecated->setIssuerDeprecated(std::string());
        REQUIRE( testVehicleDriverIdentifierDeprecated->getIssuerDeprecated() == std::string() );
    }
}
