

#include "vehicle/generated/core/vehiclepowertraintractionbatterybatteryconditioning.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryBatteryConditioning> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::setIsActive(bool IsActive)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveMutex);
    m_impl->setIsActive(IsActive);
}

bool VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::getIsActive() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isActiveMutex);
    return m_impl->getIsActive();
}
void VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::setIsOngoing(bool IsOngoing)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOngoingMutex);
    m_impl->setIsOngoing(IsOngoing);
}

bool VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::getIsOngoing() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOngoingMutex);
    return m_impl->getIsOngoing();
}
void VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::setRequestedMode(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_requestedModeMutex);
    m_impl->setRequestedMode(RequestedMode);
}

Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::getRequestedMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_requestedModeMutex);
    return m_impl->getRequestedMode();
}
void VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::setStartTime(const std::string& StartTime)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_startTimeMutex);
    m_impl->setStartTime(StartTime);
}

const std::string& VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::getStartTime() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_startTimeMutex);
    return m_impl->getStartTime();
}
void VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::setTargetTime(const std::string& TargetTime)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_targetTimeMutex);
    m_impl->setTargetTime(TargetTime);
}

const std::string& VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::getTargetTime() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_targetTimeMutex);
    return m_impl->getTargetTime();
}
void VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::setTargetTemperature(float TargetTemperature)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_targetTemperatureMutex);
    m_impl->setTargetTemperature(TargetTemperature);
}

float VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::getTargetTemperature() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_targetTemperatureMutex);
    return m_impl->getTargetTemperature();
}

IVehiclePowertrainTractionBatteryBatteryConditioningPublisher& VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}