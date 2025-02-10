

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportanyposition.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    m_impl->setSupportedInletTypes(SupportedInletTypes);
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::getSupportedInletTypes() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    return m_impl->getSupportedInletTypes();
}
void VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::setIsFlapOpen(bool IsFlapOpen)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    m_impl->setIsFlapOpen(IsFlapOpen);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::getIsFlapOpen() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    return m_impl->getIsFlapOpen();
}
void VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    m_impl->setIsChargingCableConnected(IsChargingCableConnected);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::getIsChargingCableConnected() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    return m_impl->getIsChargingCableConnected();
}
void VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    m_impl->setIsChargingCableLocked(IsChargingCableLocked);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::getIsChargingCableLocked() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    return m_impl->getIsChargingCableLocked();
}

IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher& VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}