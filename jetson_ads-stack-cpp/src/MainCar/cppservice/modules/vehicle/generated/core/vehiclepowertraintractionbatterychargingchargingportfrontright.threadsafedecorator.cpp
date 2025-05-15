

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontright.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    m_impl->setSupportedInletTypes(SupportedInletTypes);
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::getSupportedInletTypes() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    return m_impl->getSupportedInletTypes();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::setIsFlapOpen(bool IsFlapOpen)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    m_impl->setIsFlapOpen(IsFlapOpen);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::getIsFlapOpen() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    return m_impl->getIsFlapOpen();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    m_impl->setIsChargingCableConnected(IsChargingCableConnected);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::getIsChargingCableConnected() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    return m_impl->getIsChargingCableConnected();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    m_impl->setIsChargingCableLocked(IsChargingCableLocked);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::getIsChargingCableLocked() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    return m_impl->getIsChargingCableLocked();
}

IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher& VehiclePowertrainTractionBatteryChargingChargingPortFrontRightThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}