

#include "vehicle/generated/core/vehiclevehicleidentification.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleVehicleIdentificationThreadSafeDecorator::VehicleVehicleIdentificationThreadSafeDecorator(std::shared_ptr<IVehicleVehicleIdentification> impl)
    : m_impl(impl)
{
}
void VehicleVehicleIdentificationThreadSafeDecorator::setVin(const std::string& VIN)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vinMutex);
    m_impl->setVin(VIN);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getVin() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vinMutex);
    return m_impl->getVin();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setWmi(const std::string& WMI)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_wmiMutex);
    m_impl->setWmi(WMI);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getWmi() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_wmiMutex);
    return m_impl->getWmi();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setBrand(const std::string& Brand)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_brandMutex);
    m_impl->setBrand(Brand);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getBrand() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_brandMutex);
    return m_impl->getBrand();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setModel(const std::string& Model)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_modelMutex);
    m_impl->setModel(Model);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getModel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_modelMutex);
    return m_impl->getModel();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setYear(int32_t Year)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_yearMutex);
    m_impl->setYear(Year);
}

int32_t VehicleVehicleIdentificationThreadSafeDecorator::getYear() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_yearMutex);
    return m_impl->getYear();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setAcrissCode(const std::string& AcrissCode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_acrissCodeMutex);
    m_impl->setAcrissCode(AcrissCode);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getAcrissCode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_acrissCodeMutex);
    return m_impl->getAcrissCode();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setBodyType(const std::string& BodyType)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_bodyTypeMutex);
    m_impl->setBodyType(BodyType);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getBodyType() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_bodyTypeMutex);
    return m_impl->getBodyType();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setDateVehicleFirstRegistered(const std::string& DateVehicleFirstRegistered)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dateVehicleFirstRegisteredMutex);
    m_impl->setDateVehicleFirstRegistered(DateVehicleFirstRegistered);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getDateVehicleFirstRegistered() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_dateVehicleFirstRegisteredMutex);
    return m_impl->getDateVehicleFirstRegistered();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setLicensePlate(const std::string& LicensePlate)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_licensePlateMutex);
    m_impl->setLicensePlate(LicensePlate);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getLicensePlate() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_licensePlateMutex);
    return m_impl->getLicensePlate();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setMeetsEmissionStandard(const std::string& MeetsEmissionStandard)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_meetsEmissionStandardMutex);
    m_impl->setMeetsEmissionStandard(MeetsEmissionStandard);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getMeetsEmissionStandard() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_meetsEmissionStandardMutex);
    return m_impl->getMeetsEmissionStandard();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setProductionDate(const std::string& ProductionDate)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_productionDateMutex);
    m_impl->setProductionDate(ProductionDate);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getProductionDate() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_productionDateMutex);
    return m_impl->getProductionDate();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setPurchaseDate(const std::string& PurchaseDate)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_purchaseDateMutex);
    m_impl->setPurchaseDate(PurchaseDate);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getPurchaseDate() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_purchaseDateMutex);
    return m_impl->getPurchaseDate();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setVehicleModelDate(const std::string& VehicleModelDate)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleModelDateMutex);
    m_impl->setVehicleModelDate(VehicleModelDate);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getVehicleModelDate() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleModelDateMutex);
    return m_impl->getVehicleModelDate();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setVehicleConfiguration(const std::string& VehicleConfiguration)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleConfigurationMutex);
    m_impl->setVehicleConfiguration(VehicleConfiguration);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getVehicleConfiguration() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleConfigurationMutex);
    return m_impl->getVehicleConfiguration();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setVehicleSeatingCapacity(int32_t VehicleSeatingCapacity)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleSeatingCapacityMutex);
    m_impl->setVehicleSeatingCapacity(VehicleSeatingCapacity);
}

int32_t VehicleVehicleIdentificationThreadSafeDecorator::getVehicleSeatingCapacity() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleSeatingCapacityMutex);
    return m_impl->getVehicleSeatingCapacity();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setVehicleSpecialUsage(const std::string& VehicleSpecialUsage)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleSpecialUsageMutex);
    m_impl->setVehicleSpecialUsage(VehicleSpecialUsage);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getVehicleSpecialUsage() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleSpecialUsageMutex);
    return m_impl->getVehicleSpecialUsage();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setVehicleExteriorColor(const std::string& VehicleExteriorColor)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleExteriorColorMutex);
    m_impl->setVehicleExteriorColor(VehicleExteriorColor);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getVehicleExteriorColor() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleExteriorColorMutex);
    return m_impl->getVehicleExteriorColor();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setVehicleInteriorColor(const std::string& VehicleInteriorColor)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleInteriorColorMutex);
    m_impl->setVehicleInteriorColor(VehicleInteriorColor);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getVehicleInteriorColor() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleInteriorColorMutex);
    return m_impl->getVehicleInteriorColor();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setVehicleInteriorType(const std::string& VehicleInteriorType)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleInteriorTypeMutex);
    m_impl->setVehicleInteriorType(VehicleInteriorType);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getVehicleInteriorType() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleInteriorTypeMutex);
    return m_impl->getVehicleInteriorType();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setKnownVehicleDamages(const std::string& KnownVehicleDamages)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_knownVehicleDamagesMutex);
    m_impl->setKnownVehicleDamages(KnownVehicleDamages);
}

const std::string& VehicleVehicleIdentificationThreadSafeDecorator::getKnownVehicleDamages() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_knownVehicleDamagesMutex);
    return m_impl->getKnownVehicleDamages();
}
void VehicleVehicleIdentificationThreadSafeDecorator::setOptionalExtras(const std::list<std::string>& OptionalExtras)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_optionalExtrasMutex);
    m_impl->setOptionalExtras(OptionalExtras);
}

const std::list<std::string>& VehicleVehicleIdentificationThreadSafeDecorator::getOptionalExtras() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_optionalExtrasMutex);
    return m_impl->getOptionalExtras();
}

IVehicleVehicleIdentificationPublisher& VehicleVehicleIdentificationThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}