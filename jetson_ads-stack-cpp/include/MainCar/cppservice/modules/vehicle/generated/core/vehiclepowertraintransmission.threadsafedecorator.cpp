

#include "vehicle/generated/core/vehiclepowertraintransmission.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTransmissionThreadSafeDecorator::VehiclePowertrainTransmissionThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTransmission> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setType(Vehicle_Powertrain_Transmission_Type_ValueEnum Type)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_typeMutex);
    m_impl->setType(Type);
}

Vehicle_Powertrain_Transmission_Type_ValueEnum VehiclePowertrainTransmissionThreadSafeDecorator::getType() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_typeMutex);
    return m_impl->getType();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setGearCount(int32_t GearCount)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_gearCountMutex);
    m_impl->setGearCount(GearCount);
}

int32_t VehiclePowertrainTransmissionThreadSafeDecorator::getGearCount() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_gearCountMutex);
    return m_impl->getGearCount();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setDriveType(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_driveTypeMutex);
    m_impl->setDriveType(DriveType);
}

Vehicle_Powertrain_Transmission_DriveType_ValueEnum VehiclePowertrainTransmissionThreadSafeDecorator::getDriveType() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_driveTypeMutex);
    return m_impl->getDriveType();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setTravelledDistance(float TravelledDistance)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_travelledDistanceMutex);
    m_impl->setTravelledDistance(TravelledDistance);
}

float VehiclePowertrainTransmissionThreadSafeDecorator::getTravelledDistance() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_travelledDistanceMutex);
    return m_impl->getTravelledDistance();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setCurrentGear(int32_t CurrentGear)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentGearMutex);
    m_impl->setCurrentGear(CurrentGear);
}

int32_t VehiclePowertrainTransmissionThreadSafeDecorator::getCurrentGear() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_currentGearMutex);
    return m_impl->getCurrentGear();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setSelectedGear(int32_t SelectedGear)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_selectedGearMutex);
    m_impl->setSelectedGear(SelectedGear);
}

int32_t VehiclePowertrainTransmissionThreadSafeDecorator::getSelectedGear() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_selectedGearMutex);
    return m_impl->getSelectedGear();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setIsParkLockEngaged(bool IsParkLockEngaged)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isParkLockEngagedMutex);
    m_impl->setIsParkLockEngaged(IsParkLockEngaged);
}

bool VehiclePowertrainTransmissionThreadSafeDecorator::getIsParkLockEngaged() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isParkLockEngagedMutex);
    return m_impl->getIsParkLockEngaged();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setIsLowRangeEngaged(bool IsLowRangeEngaged)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isLowRangeEngagedMutex);
    m_impl->setIsLowRangeEngaged(IsLowRangeEngaged);
}

bool VehiclePowertrainTransmissionThreadSafeDecorator::getIsLowRangeEngaged() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isLowRangeEngagedMutex);
    return m_impl->getIsLowRangeEngaged();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setIsElectricalPowertrainEngaged(bool IsElectricalPowertrainEngaged)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isElectricalPowertrainEngagedMutex);
    m_impl->setIsElectricalPowertrainEngaged(IsElectricalPowertrainEngaged);
}

bool VehiclePowertrainTransmissionThreadSafeDecorator::getIsElectricalPowertrainEngaged() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isElectricalPowertrainEngagedMutex);
    return m_impl->getIsElectricalPowertrainEngaged();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setPerformanceMode(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_performanceModeMutex);
    m_impl->setPerformanceMode(PerformanceMode);
}

Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum VehiclePowertrainTransmissionThreadSafeDecorator::getPerformanceMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_performanceModeMutex);
    return m_impl->getPerformanceMode();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setGearChangeMode(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_gearChangeModeMutex);
    m_impl->setGearChangeMode(GearChangeMode);
}

Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum VehiclePowertrainTransmissionThreadSafeDecorator::getGearChangeMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_gearChangeModeMutex);
    return m_impl->getGearChangeMode();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setTemperature(float Temperature)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    m_impl->setTemperature(Temperature);
}

float VehiclePowertrainTransmissionThreadSafeDecorator::getTemperature() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    return m_impl->getTemperature();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setClutchEngagement(float ClutchEngagement)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_clutchEngagementMutex);
    m_impl->setClutchEngagement(ClutchEngagement);
}

float VehiclePowertrainTransmissionThreadSafeDecorator::getClutchEngagement() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_clutchEngagementMutex);
    return m_impl->getClutchEngagement();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setClutchWear(int32_t ClutchWear)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_clutchWearMutex);
    m_impl->setClutchWear(ClutchWear);
}

int32_t VehiclePowertrainTransmissionThreadSafeDecorator::getClutchWear() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_clutchWearMutex);
    return m_impl->getClutchWear();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setDiffLockFrontEngagement(float DiffLockFrontEngagement)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_diffLockFrontEngagementMutex);
    m_impl->setDiffLockFrontEngagement(DiffLockFrontEngagement);
}

float VehiclePowertrainTransmissionThreadSafeDecorator::getDiffLockFrontEngagement() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_diffLockFrontEngagementMutex);
    return m_impl->getDiffLockFrontEngagement();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setDiffLockRearEngagement(float DiffLockRearEngagement)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_diffLockRearEngagementMutex);
    m_impl->setDiffLockRearEngagement(DiffLockRearEngagement);
}

float VehiclePowertrainTransmissionThreadSafeDecorator::getDiffLockRearEngagement() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_diffLockRearEngagementMutex);
    return m_impl->getDiffLockRearEngagement();
}
void VehiclePowertrainTransmissionThreadSafeDecorator::setTorqueDistribution(float TorqueDistribution)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionMutex);
    m_impl->setTorqueDistribution(TorqueDistribution);
}

float VehiclePowertrainTransmissionThreadSafeDecorator::getTorqueDistribution() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueDistributionMutex);
    return m_impl->getTorqueDistribution();
}

IVehiclePowertrainTransmissionPublisher& VehiclePowertrainTransmissionThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}