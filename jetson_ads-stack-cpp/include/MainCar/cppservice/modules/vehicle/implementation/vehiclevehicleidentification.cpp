

#include "vehicle/implementation/vehiclevehicleidentification.h"
#include "vehicle/generated/core/vehiclevehicleidentification.publisher.h"
#include "vehicle/generated/core/vehiclevehicleidentification.data.h"

using namespace Cpp::Vehicle;

VehicleVehicleIdentification::VehicleVehicleIdentification()
    : m_publisher(std::make_unique<VehicleVehicleIdentificationPublisher>())
{
}
VehicleVehicleIdentification::~VehicleVehicleIdentification()
{
}

void VehicleVehicleIdentification::setVin(const std::string& VIN)
{
    if (m_data.m_VIN != VIN) {
        m_data.m_VIN = VIN;
        m_publisher->publishVinChanged(VIN);
    }
}

const std::string& VehicleVehicleIdentification::getVin() const
{
    return m_data.m_VIN;
}

void VehicleVehicleIdentification::setWmi(const std::string& WMI)
{
    if (m_data.m_WMI != WMI) {
        m_data.m_WMI = WMI;
        m_publisher->publishWmiChanged(WMI);
    }
}

const std::string& VehicleVehicleIdentification::getWmi() const
{
    return m_data.m_WMI;
}

void VehicleVehicleIdentification::setBrand(const std::string& Brand)
{
    if (m_data.m_Brand != Brand) {
        m_data.m_Brand = Brand;
        m_publisher->publishBrandChanged(Brand);
    }
}

const std::string& VehicleVehicleIdentification::getBrand() const
{
    return m_data.m_Brand;
}

void VehicleVehicleIdentification::setModel(const std::string& Model)
{
    if (m_data.m_Model != Model) {
        m_data.m_Model = Model;
        m_publisher->publishModelChanged(Model);
    }
}

const std::string& VehicleVehicleIdentification::getModel() const
{
    return m_data.m_Model;
}

void VehicleVehicleIdentification::setYear(int32_t Year)
{
    if (m_data.m_Year != Year) {
        m_data.m_Year = Year;
        m_publisher->publishYearChanged(Year);
    }
}

int32_t VehicleVehicleIdentification::getYear() const
{
    return m_data.m_Year;
}

void VehicleVehicleIdentification::setAcrissCode(const std::string& AcrissCode)
{
    if (m_data.m_AcrissCode != AcrissCode) {
        m_data.m_AcrissCode = AcrissCode;
        m_publisher->publishAcrissCodeChanged(AcrissCode);
    }
}

const std::string& VehicleVehicleIdentification::getAcrissCode() const
{
    return m_data.m_AcrissCode;
}

void VehicleVehicleIdentification::setBodyType(const std::string& BodyType)
{
    if (m_data.m_BodyType != BodyType) {
        m_data.m_BodyType = BodyType;
        m_publisher->publishBodyTypeChanged(BodyType);
    }
}

const std::string& VehicleVehicleIdentification::getBodyType() const
{
    return m_data.m_BodyType;
}

void VehicleVehicleIdentification::setDateVehicleFirstRegistered(const std::string& DateVehicleFirstRegistered)
{
    if (m_data.m_DateVehicleFirstRegistered != DateVehicleFirstRegistered) {
        m_data.m_DateVehicleFirstRegistered = DateVehicleFirstRegistered;
        m_publisher->publishDateVehicleFirstRegisteredChanged(DateVehicleFirstRegistered);
    }
}

const std::string& VehicleVehicleIdentification::getDateVehicleFirstRegistered() const
{
    return m_data.m_DateVehicleFirstRegistered;
}

void VehicleVehicleIdentification::setLicensePlate(const std::string& LicensePlate)
{
    if (m_data.m_LicensePlate != LicensePlate) {
        m_data.m_LicensePlate = LicensePlate;
        m_publisher->publishLicensePlateChanged(LicensePlate);
    }
}

const std::string& VehicleVehicleIdentification::getLicensePlate() const
{
    return m_data.m_LicensePlate;
}

void VehicleVehicleIdentification::setMeetsEmissionStandard(const std::string& MeetsEmissionStandard)
{
    if (m_data.m_MeetsEmissionStandard != MeetsEmissionStandard) {
        m_data.m_MeetsEmissionStandard = MeetsEmissionStandard;
        m_publisher->publishMeetsEmissionStandardChanged(MeetsEmissionStandard);
    }
}

const std::string& VehicleVehicleIdentification::getMeetsEmissionStandard() const
{
    return m_data.m_MeetsEmissionStandard;
}

void VehicleVehicleIdentification::setProductionDate(const std::string& ProductionDate)
{
    if (m_data.m_ProductionDate != ProductionDate) {
        m_data.m_ProductionDate = ProductionDate;
        m_publisher->publishProductionDateChanged(ProductionDate);
    }
}

const std::string& VehicleVehicleIdentification::getProductionDate() const
{
    return m_data.m_ProductionDate;
}

void VehicleVehicleIdentification::setPurchaseDate(const std::string& PurchaseDate)
{
    if (m_data.m_PurchaseDate != PurchaseDate) {
        m_data.m_PurchaseDate = PurchaseDate;
        m_publisher->publishPurchaseDateChanged(PurchaseDate);
    }
}

const std::string& VehicleVehicleIdentification::getPurchaseDate() const
{
    return m_data.m_PurchaseDate;
}

void VehicleVehicleIdentification::setVehicleModelDate(const std::string& VehicleModelDate)
{
    if (m_data.m_VehicleModelDate != VehicleModelDate) {
        m_data.m_VehicleModelDate = VehicleModelDate;
        m_publisher->publishVehicleModelDateChanged(VehicleModelDate);
    }
}

const std::string& VehicleVehicleIdentification::getVehicleModelDate() const
{
    return m_data.m_VehicleModelDate;
}

void VehicleVehicleIdentification::setVehicleConfiguration(const std::string& VehicleConfiguration)
{
    if (m_data.m_VehicleConfiguration != VehicleConfiguration) {
        m_data.m_VehicleConfiguration = VehicleConfiguration;
        m_publisher->publishVehicleConfigurationChanged(VehicleConfiguration);
    }
}

const std::string& VehicleVehicleIdentification::getVehicleConfiguration() const
{
    return m_data.m_VehicleConfiguration;
}

void VehicleVehicleIdentification::setVehicleSeatingCapacity(int32_t VehicleSeatingCapacity)
{
    if (m_data.m_VehicleSeatingCapacity != VehicleSeatingCapacity) {
        m_data.m_VehicleSeatingCapacity = VehicleSeatingCapacity;
        m_publisher->publishVehicleSeatingCapacityChanged(VehicleSeatingCapacity);
    }
}

int32_t VehicleVehicleIdentification::getVehicleSeatingCapacity() const
{
    return m_data.m_VehicleSeatingCapacity;
}

void VehicleVehicleIdentification::setVehicleSpecialUsage(const std::string& VehicleSpecialUsage)
{
    if (m_data.m_VehicleSpecialUsage != VehicleSpecialUsage) {
        m_data.m_VehicleSpecialUsage = VehicleSpecialUsage;
        m_publisher->publishVehicleSpecialUsageChanged(VehicleSpecialUsage);
    }
}

const std::string& VehicleVehicleIdentification::getVehicleSpecialUsage() const
{
    return m_data.m_VehicleSpecialUsage;
}

void VehicleVehicleIdentification::setVehicleExteriorColor(const std::string& VehicleExteriorColor)
{
    if (m_data.m_VehicleExteriorColor != VehicleExteriorColor) {
        m_data.m_VehicleExteriorColor = VehicleExteriorColor;
        m_publisher->publishVehicleExteriorColorChanged(VehicleExteriorColor);
    }
}

const std::string& VehicleVehicleIdentification::getVehicleExteriorColor() const
{
    return m_data.m_VehicleExteriorColor;
}

void VehicleVehicleIdentification::setVehicleInteriorColor(const std::string& VehicleInteriorColor)
{
    if (m_data.m_VehicleInteriorColor != VehicleInteriorColor) {
        m_data.m_VehicleInteriorColor = VehicleInteriorColor;
        m_publisher->publishVehicleInteriorColorChanged(VehicleInteriorColor);
    }
}

const std::string& VehicleVehicleIdentification::getVehicleInteriorColor() const
{
    return m_data.m_VehicleInteriorColor;
}

void VehicleVehicleIdentification::setVehicleInteriorType(const std::string& VehicleInteriorType)
{
    if (m_data.m_VehicleInteriorType != VehicleInteriorType) {
        m_data.m_VehicleInteriorType = VehicleInteriorType;
        m_publisher->publishVehicleInteriorTypeChanged(VehicleInteriorType);
    }
}

const std::string& VehicleVehicleIdentification::getVehicleInteriorType() const
{
    return m_data.m_VehicleInteriorType;
}

void VehicleVehicleIdentification::setKnownVehicleDamages(const std::string& KnownVehicleDamages)
{
    if (m_data.m_KnownVehicleDamages != KnownVehicleDamages) {
        m_data.m_KnownVehicleDamages = KnownVehicleDamages;
        m_publisher->publishKnownVehicleDamagesChanged(KnownVehicleDamages);
    }
}

const std::string& VehicleVehicleIdentification::getKnownVehicleDamages() const
{
    return m_data.m_KnownVehicleDamages;
}

void VehicleVehicleIdentification::setOptionalExtras(const std::list<std::string>& OptionalExtras)
{
    if (m_data.m_OptionalExtras != OptionalExtras) {
        m_data.m_OptionalExtras = OptionalExtras;
        m_publisher->publishOptionalExtrasChanged(OptionalExtras);
    }
}

const std::list<std::string>& VehicleVehicleIdentification::getOptionalExtras() const
{
    return m_data.m_OptionalExtras;
}

IVehicleVehicleIdentificationPublisher& VehicleVehicleIdentification::_getPublisher() const
{
    return *m_publisher;
}
