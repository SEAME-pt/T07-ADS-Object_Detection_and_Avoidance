

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportrearleft.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearLeft> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    m_impl->setSupportedInletTypes(SupportedInletTypes);
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::getSupportedInletTypes() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    return m_impl->getSupportedInletTypes();
}
void VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::setIsFlapOpen(bool IsFlapOpen)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    m_impl->setIsFlapOpen(IsFlapOpen);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::getIsFlapOpen() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    return m_impl->getIsFlapOpen();
}
void VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    m_impl->setIsChargingCableConnected(IsChargingCableConnected);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::getIsChargingCableConnected() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    return m_impl->getIsChargingCableConnected();
}
void VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    m_impl->setIsChargingCableLocked(IsChargingCableLocked);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::getIsChargingCableLocked() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    return m_impl->getIsChargingCableLocked();
}

IVehiclePowertrainTractionBatteryChargingChargingPortRearLeftPublisher& VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}