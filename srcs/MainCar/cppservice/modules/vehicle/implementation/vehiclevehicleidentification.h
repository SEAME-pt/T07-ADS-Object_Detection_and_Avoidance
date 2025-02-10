
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclevehicleidentification.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_VehicleIdentification implementation.
*/
class CPP_VEHICLE_EXPORT VehicleVehicleIdentification : public IVehicleVehicleIdentification
{
public:
    explicit VehicleVehicleIdentification();
    ~VehicleVehicleIdentification();
public:
    /**
    * VIN 17-character Vehicle Identification Number (VIN) as defined by ISO 3779.
    */
    void setVin(const std::string& VIN) override;
    const std::string& getVin() const override;
    
    /**
    * WMI 3-character World Manufacturer Identification (WMI) as defined by ISO 3780.
    */
    void setWmi(const std::string& WMI) override;
    const std::string& getWmi() const override;
    
    /**
    * Brand Vehicle brand or manufacturer.
    */
    void setBrand(const std::string& Brand) override;
    const std::string& getBrand() const override;
    
    /**
    * Model Vehicle model.
    */
    void setModel(const std::string& Model) override;
    const std::string& getModel() const override;
    
    /**
    * Year Model year of the vehicle.
    */
    void setYear(int32_t Year) override;
    int32_t getYear() const override;
    
    /**
    * AcrissCode The ACRISS Car Classification Code is a code used by many car rental companies.
    */
    void setAcrissCode(const std::string& AcrissCode) override;
    const std::string& getAcrissCode() const override;
    
    /**
    * BodyType Indicates the design and body style of the vehicle (e.g. station wagon, hatchback, etc.).
    */
    void setBodyType(const std::string& BodyType) override;
    const std::string& getBodyType() const override;
    
    /**
    * DateVehicleFirstRegistered The date in ISO 8601 format of the first registration of the vehicle with the respective public authorities.
    */
    void setDateVehicleFirstRegistered(const std::string& DateVehicleFirstRegistered) override;
    const std::string& getDateVehicleFirstRegistered() const override;
    
    /**
    * LicensePlate The license plate of the vehicle.
    */
    void setLicensePlate(const std::string& LicensePlate) override;
    const std::string& getLicensePlate() const override;
    
    /**
    * MeetsEmissionStandard Indicates that the vehicle meets the respective emission standard.
    */
    void setMeetsEmissionStandard(const std::string& MeetsEmissionStandard) override;
    const std::string& getMeetsEmissionStandard() const override;
    
    /**
    * ProductionDate The date in ISO 8601 format of production of the item, e.g. vehicle.
    */
    void setProductionDate(const std::string& ProductionDate) override;
    const std::string& getProductionDate() const override;
    
    /**
    * PurchaseDate The date in ISO 8601 format of the item e.g. vehicle was purchased by the current owner.
    */
    void setPurchaseDate(const std::string& PurchaseDate) override;
    const std::string& getPurchaseDate() const override;
    
    /**
    * VehicleModelDate The release date in ISO 8601 format of a vehicle model (often used to differentiate versions of the same make and model).
    */
    void setVehicleModelDate(const std::string& VehicleModelDate) override;
    const std::string& getVehicleModelDate() const override;
    
    /**
    * VehicleConfiguration A short text indicating the configuration of the vehicle, e.g. '5dr hatchback ST 2.5 MT 225 hp' or 'limited edition'.
    */
    void setVehicleConfiguration(const std::string& VehicleConfiguration) override;
    const std::string& getVehicleConfiguration() const override;
    
    /**
    * VehicleSeatingCapacity The number of passengers that can be seated in the vehicle, both in terms of the physical space available, and in terms of limitations set by law.
    */
    void setVehicleSeatingCapacity(int32_t VehicleSeatingCapacity) override;
    int32_t getVehicleSeatingCapacity() const override;
    
    /**
    * VehicleSpecialUsage Indicates whether the vehicle has been used for special purposes, like commercial rental, driving school.
    */
    void setVehicleSpecialUsage(const std::string& VehicleSpecialUsage) override;
    const std::string& getVehicleSpecialUsage() const override;
    
    /**
    * VehicleExteriorColor The main color of the exterior within the basic color palette (eg. red, blue, black, white, ...).
    */
    void setVehicleExteriorColor(const std::string& VehicleExteriorColor) override;
    const std::string& getVehicleExteriorColor() const override;
    
    /**
    * VehicleInteriorColor The color or color combination of the interior of the vehicle.
    */
    void setVehicleInteriorColor(const std::string& VehicleInteriorColor) override;
    const std::string& getVehicleInteriorColor() const override;
    
    /**
    * VehicleInteriorType The type or material of the interior of the vehicle (e.g. synthetic fabric, leather, wood, etc.).
    */
    void setVehicleInteriorType(const std::string& VehicleInteriorType) override;
    const std::string& getVehicleInteriorType() const override;
    
    /**
    * KnownVehicleDamages A textual description of known damages, both repaired and unrepaired.
    */
    void setKnownVehicleDamages(const std::string& KnownVehicleDamages) override;
    const std::string& getKnownVehicleDamages() const override;
    
    /**
    * OptionalExtras Optional extras refers to all car equipment options that are not installed as standard by the manufacturer.
    */
    void setOptionalExtras(const std::list<std::string>& OptionalExtras) override;
    const std::list<std::string>& getOptionalExtras() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_VehicleIdentification changes and signal emission.
    * @return The publisher for Vehicle_VehicleIdentification.
    */
    IVehicleVehicleIdentificationPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_VehicleIdentification. */
    std::unique_ptr<IVehicleVehicleIdentificationPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_VehicleIdentification. */
    VehicleVehicleIdentificationData m_data;
};
} // namespace Vehicle
} // namespace Cpp
