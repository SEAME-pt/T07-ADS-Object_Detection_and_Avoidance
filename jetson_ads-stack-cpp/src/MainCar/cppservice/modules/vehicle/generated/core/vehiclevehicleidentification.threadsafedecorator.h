
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleVehicleIdentificationThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_VehicleIdentification interface.
* @see Vehicle_VehicleIdentification
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleVehicleIdentification> testVehicleVehicleIdentification = std::make_unique<VehicleVehicleIdentificationThreadSafeDecorator>(std::make_shared<VehicleVehicleIdentification>());

// Thread safe access
auto vin = testVehicleVehicleIdentification->getVin();
testVehicleVehicleIdentification->setVin(std::string());
auto wmi = testVehicleVehicleIdentification->getWmi();
testVehicleVehicleIdentification->setWmi(std::string());
auto brand = testVehicleVehicleIdentification->getBrand();
testVehicleVehicleIdentification->setBrand(std::string());
auto model = testVehicleVehicleIdentification->getModel();
testVehicleVehicleIdentification->setModel(std::string());
auto year = testVehicleVehicleIdentification->getYear();
testVehicleVehicleIdentification->setYear(0);
auto acrissCode = testVehicleVehicleIdentification->getAcrissCode();
testVehicleVehicleIdentification->setAcrissCode(std::string());
auto bodyType = testVehicleVehicleIdentification->getBodyType();
testVehicleVehicleIdentification->setBodyType(std::string());
auto dateVehicleFirstRegistered = testVehicleVehicleIdentification->getDateVehicleFirstRegistered();
testVehicleVehicleIdentification->setDateVehicleFirstRegistered(std::string());
auto licensePlate = testVehicleVehicleIdentification->getLicensePlate();
testVehicleVehicleIdentification->setLicensePlate(std::string());
auto meetsEmissionStandard = testVehicleVehicleIdentification->getMeetsEmissionStandard();
testVehicleVehicleIdentification->setMeetsEmissionStandard(std::string());
auto productionDate = testVehicleVehicleIdentification->getProductionDate();
testVehicleVehicleIdentification->setProductionDate(std::string());
auto purchaseDate = testVehicleVehicleIdentification->getPurchaseDate();
testVehicleVehicleIdentification->setPurchaseDate(std::string());
auto vehicleModelDate = testVehicleVehicleIdentification->getVehicleModelDate();
testVehicleVehicleIdentification->setVehicleModelDate(std::string());
auto vehicleConfiguration = testVehicleVehicleIdentification->getVehicleConfiguration();
testVehicleVehicleIdentification->setVehicleConfiguration(std::string());
auto vehicleSeatingCapacity = testVehicleVehicleIdentification->getVehicleSeatingCapacity();
testVehicleVehicleIdentification->setVehicleSeatingCapacity(0);
auto vehicleSpecialUsage = testVehicleVehicleIdentification->getVehicleSpecialUsage();
testVehicleVehicleIdentification->setVehicleSpecialUsage(std::string());
auto vehicleExteriorColor = testVehicleVehicleIdentification->getVehicleExteriorColor();
testVehicleVehicleIdentification->setVehicleExteriorColor(std::string());
auto vehicleInteriorColor = testVehicleVehicleIdentification->getVehicleInteriorColor();
testVehicleVehicleIdentification->setVehicleInteriorColor(std::string());
auto vehicleInteriorType = testVehicleVehicleIdentification->getVehicleInteriorType();
testVehicleVehicleIdentification->setVehicleInteriorType(std::string());
auto knownVehicleDamages = testVehicleVehicleIdentification->getKnownVehicleDamages();
testVehicleVehicleIdentification->setKnownVehicleDamages(std::string());
auto optionalExtras = testVehicleVehicleIdentification->getOptionalExtras();
testVehicleVehicleIdentification->setOptionalExtras(std::list<std::string>());
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleVehicleIdentificationThreadSafeDecorator : public IVehicleVehicleIdentification
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_VehicleIdentification object to make thread safe.
    */
    explicit VehicleVehicleIdentificationThreadSafeDecorator(std::shared_ptr<IVehicleVehicleIdentification> impl);

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setVin(const std::string& VIN) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getVin() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setWmi(const std::string& WMI) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getWmi() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setBrand(const std::string& Brand) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getBrand() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setModel(const std::string& Model) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getModel() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setYear(int32_t Year) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    int32_t getYear() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setAcrissCode(const std::string& AcrissCode) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getAcrissCode() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setBodyType(const std::string& BodyType) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getBodyType() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setDateVehicleFirstRegistered(const std::string& DateVehicleFirstRegistered) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getDateVehicleFirstRegistered() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setLicensePlate(const std::string& LicensePlate) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getLicensePlate() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setMeetsEmissionStandard(const std::string& MeetsEmissionStandard) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getMeetsEmissionStandard() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setProductionDate(const std::string& ProductionDate) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getProductionDate() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setPurchaseDate(const std::string& PurchaseDate) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getPurchaseDate() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setVehicleModelDate(const std::string& VehicleModelDate) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getVehicleModelDate() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setVehicleConfiguration(const std::string& VehicleConfiguration) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getVehicleConfiguration() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setVehicleSeatingCapacity(int32_t VehicleSeatingCapacity) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    int32_t getVehicleSeatingCapacity() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setVehicleSpecialUsage(const std::string& VehicleSpecialUsage) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getVehicleSpecialUsage() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setVehicleExteriorColor(const std::string& VehicleExteriorColor) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getVehicleExteriorColor() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setVehicleInteriorColor(const std::string& VehicleInteriorColor) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getVehicleInteriorColor() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setVehicleInteriorType(const std::string& VehicleInteriorType) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getVehicleInteriorType() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setKnownVehicleDamages(const std::string& KnownVehicleDamages) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::string& getKnownVehicleDamages() const override;

    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    void setOptionalExtras(const std::list<std::string>& OptionalExtras) override;
    /** Guards and forwards call to Vehicle_VehicleIdentification implementation. */
    const std::list<std::string>& getOptionalExtras() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_VehicleIdentification changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_VehicleIdentification.
    */
    IVehicleVehicleIdentificationPublisher& _getPublisher() const override;
private:
    /** The Vehicle_VehicleIdentification object which is guarded */
    std::shared_ptr<IVehicleVehicleIdentification> m_impl;
    // Mutex for vin property
    mutable std::shared_timed_mutex m_vinMutex;
    // Mutex for wmi property
    mutable std::shared_timed_mutex m_wmiMutex;
    // Mutex for brand property
    mutable std::shared_timed_mutex m_brandMutex;
    // Mutex for model property
    mutable std::shared_timed_mutex m_modelMutex;
    // Mutex for year property
    mutable std::shared_timed_mutex m_yearMutex;
    // Mutex for acrissCode property
    mutable std::shared_timed_mutex m_acrissCodeMutex;
    // Mutex for bodyType property
    mutable std::shared_timed_mutex m_bodyTypeMutex;
    // Mutex for dateVehicleFirstRegistered property
    mutable std::shared_timed_mutex m_dateVehicleFirstRegisteredMutex;
    // Mutex for licensePlate property
    mutable std::shared_timed_mutex m_licensePlateMutex;
    // Mutex for meetsEmissionStandard property
    mutable std::shared_timed_mutex m_meetsEmissionStandardMutex;
    // Mutex for productionDate property
    mutable std::shared_timed_mutex m_productionDateMutex;
    // Mutex for purchaseDate property
    mutable std::shared_timed_mutex m_purchaseDateMutex;
    // Mutex for vehicleModelDate property
    mutable std::shared_timed_mutex m_vehicleModelDateMutex;
    // Mutex for vehicleConfiguration property
    mutable std::shared_timed_mutex m_vehicleConfigurationMutex;
    // Mutex for vehicleSeatingCapacity property
    mutable std::shared_timed_mutex m_vehicleSeatingCapacityMutex;
    // Mutex for vehicleSpecialUsage property
    mutable std::shared_timed_mutex m_vehicleSpecialUsageMutex;
    // Mutex for vehicleExteriorColor property
    mutable std::shared_timed_mutex m_vehicleExteriorColorMutex;
    // Mutex for vehicleInteriorColor property
    mutable std::shared_timed_mutex m_vehicleInteriorColorMutex;
    // Mutex for vehicleInteriorType property
    mutable std::shared_timed_mutex m_vehicleInteriorTypeMutex;
    // Mutex for knownVehicleDamages property
    mutable std::shared_timed_mutex m_knownVehicleDamagesMutex;
    // Mutex for optionalExtras property
    mutable std::shared_timed_mutex m_optionalExtrasMutex;
};
} // namespace Vehicle
} // namespace Cpp
