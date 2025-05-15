#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_VehicleIdentification. Stores all the properties.
*/
struct VehicleVehicleIdentificationData
{
    std::string m_VIN {std::string()};
    std::string m_WMI {std::string()};
    std::string m_Brand {std::string()};
    std::string m_Model {std::string()};
    int32_t m_Year {0};
    std::string m_AcrissCode {std::string()};
    std::string m_BodyType {std::string()};
    std::string m_DateVehicleFirstRegistered {std::string()};
    std::string m_LicensePlate {std::string()};
    std::string m_MeetsEmissionStandard {std::string()};
    std::string m_ProductionDate {std::string()};
    std::string m_PurchaseDate {std::string()};
    std::string m_VehicleModelDate {std::string()};
    std::string m_VehicleConfiguration {std::string()};
    int32_t m_VehicleSeatingCapacity {0};
    std::string m_VehicleSpecialUsage {std::string()};
    std::string m_VehicleExteriorColor {std::string()};
    std::string m_VehicleInteriorColor {std::string()};
    std::string m_VehicleInteriorType {std::string()};
    std::string m_KnownVehicleDamages {std::string()};
    std::list<std::string> m_OptionalExtras {std::list<std::string>()};
};

}
}