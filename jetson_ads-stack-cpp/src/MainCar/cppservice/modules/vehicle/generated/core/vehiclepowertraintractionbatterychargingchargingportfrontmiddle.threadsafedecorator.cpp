

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontmiddle.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    m_impl->setSupportedInletTypes(SupportedInletTypes);
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::getSupportedInletTypes() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    return m_impl->getSupportedInletTypes();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::setIsFlapOpen(bool IsFlapOpen)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    m_impl->setIsFlapOpen(IsFlapOpen);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::getIsFlapOpen() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    return m_impl->getIsFlapOpen();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    m_impl->setIsChargingCableConnected(IsChargingCableConnected);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::getIsChargingCableConnected() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    return m_impl->getIsChargingCableConnected();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    m_impl->setIsChargingCableLocked(IsChargingCableLocked);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::getIsChargingCableLocked() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    return m_impl->getIsChargingCableLocked();
}

IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher& VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}