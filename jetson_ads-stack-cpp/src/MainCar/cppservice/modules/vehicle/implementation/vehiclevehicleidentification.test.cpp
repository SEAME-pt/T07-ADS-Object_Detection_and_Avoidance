#include <memory>
#include "catch2/catch.hpp"
#include "vehicle/implementation/vehiclevehicleidentification.h"

using namespace Cpp::Vehicle;
TEST_CASE("Testing VehicleVehicleIdentification", "[VehicleVehicleIdentification]"){
    std::unique_ptr<IVehicleVehicleIdentification> testVehicleVehicleIdentification = std::make_unique<VehicleVehicleIdentification>();
    // setup your test
    SECTION("Test property VIN") {
        // Do implement test here
        testVehicleVehicleIdentification->setVin(std::string());
        REQUIRE( testVehicleVehicleIdentification->getVin() == std::string() );
    }
    SECTION("Test property WMI") {
        // Do implement test here
        testVehicleVehicleIdentification->setWmi(std::string());
        REQUIRE( testVehicleVehicleIdentification->getWmi() == std::string() );
    }
    SECTION("Test property Brand") {
        // Do implement test here
        testVehicleVehicleIdentification->setBrand(std::string());
        REQUIRE( testVehicleVehicleIdentification->getBrand() == std::string() );
    }
    SECTION("Test property Model") {
        // Do implement test here
        testVehicleVehicleIdentification->setModel(std::string());
        REQUIRE( testVehicleVehicleIdentification->getModel() == std::string() );
    }
    SECTION("Test property Year") {
        // Do implement test here
        testVehicleVehicleIdentification->setYear(0);
        REQUIRE( testVehicleVehicleIdentification->getYear() == 0 );
    }
    SECTION("Test property AcrissCode") {
        // Do implement test here
        testVehicleVehicleIdentification->setAcrissCode(std::string());
        REQUIRE( testVehicleVehicleIdentification->getAcrissCode() == std::string() );
    }
    SECTION("Test property BodyType") {
        // Do implement test here
        testVehicleVehicleIdentification->setBodyType(std::string());
        REQUIRE( testVehicleVehicleIdentification->getBodyType() == std::string() );
    }
    SECTION("Test property DateVehicleFirstRegistered") {
        // Do implement test here
        testVehicleVehicleIdentification->setDateVehicleFirstRegistered(std::string());
        REQUIRE( testVehicleVehicleIdentification->getDateVehicleFirstRegistered() == std::string() );
    }
    SECTION("Test property LicensePlate") {
        // Do implement test here
        testVehicleVehicleIdentification->setLicensePlate(std::string());
        REQUIRE( testVehicleVehicleIdentification->getLicensePlate() == std::string() );
    }
    SECTION("Test property MeetsEmissionStandard") {
        // Do implement test here
        testVehicleVehicleIdentification->setMeetsEmissionStandard(std::string());
        REQUIRE( testVehicleVehicleIdentification->getMeetsEmissionStandard() == std::string() );
    }
    SECTION("Test property ProductionDate") {
        // Do implement test here
        testVehicleVehicleIdentification->setProductionDate(std::string());
        REQUIRE( testVehicleVehicleIdentification->getProductionDate() == std::string() );
    }
    SECTION("Test property PurchaseDate") {
        // Do implement test here
        testVehicleVehicleIdentification->setPurchaseDate(std::string());
        REQUIRE( testVehicleVehicleIdentification->getPurchaseDate() == std::string() );
    }
    SECTION("Test property VehicleModelDate") {
        // Do implement test here
        testVehicleVehicleIdentification->setVehicleModelDate(std::string());
        REQUIRE( testVehicleVehicleIdentification->getVehicleModelDate() == std::string() );
    }
    SECTION("Test property VehicleConfiguration") {
        // Do implement test here
        testVehicleVehicleIdentification->setVehicleConfiguration(std::string());
        REQUIRE( testVehicleVehicleIdentification->getVehicleConfiguration() == std::string() );
    }
    SECTION("Test property VehicleSeatingCapacity") {
        // Do implement test here
        testVehicleVehicleIdentification->setVehicleSeatingCapacity(0);
        REQUIRE( testVehicleVehicleIdentification->getVehicleSeatingCapacity() == 0 );
    }
    SECTION("Test property VehicleSpecialUsage") {
        // Do implement test here
        testVehicleVehicleIdentification->setVehicleSpecialUsage(std::string());
        REQUIRE( testVehicleVehicleIdentification->getVehicleSpecialUsage() == std::string() );
    }
    SECTION("Test property VehicleExteriorColor") {
        // Do implement test here
        testVehicleVehicleIdentification->setVehicleExteriorColor(std::string());
        REQUIRE( testVehicleVehicleIdentification->getVehicleExteriorColor() == std::string() );
    }
    SECTION("Test property VehicleInteriorColor") {
        // Do implement test here
        testVehicleVehicleIdentification->setVehicleInteriorColor(std::string());
        REQUIRE( testVehicleVehicleIdentification->getVehicleInteriorColor() == std::string() );
    }
    SECTION("Test property VehicleInteriorType") {
        // Do implement test here
        testVehicleVehicleIdentification->setVehicleInteriorType(std::string());
        REQUIRE( testVehicleVehicleIdentification->getVehicleInteriorType() == std::string() );
    }
    SECTION("Test property KnownVehicleDamages") {
        // Do implement test here
        testVehicleVehicleIdentification->setKnownVehicleDamages(std::string());
        REQUIRE( testVehicleVehicleIdentification->getKnownVehicleDamages() == std::string() );
    }
    SECTION("Test property OptionalExtras") {
        // Do implement test here
        testVehicleVehicleIdentification->setOptionalExtras(std::list<std::string>());
        REQUIRE( testVehicleVehicleIdentification->getOptionalExtras() == std::list<std::string>() );
    }
}
