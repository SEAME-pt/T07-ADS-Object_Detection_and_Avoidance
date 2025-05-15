

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontleft.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    m_impl->setSupportedInletTypes(SupportedInletTypes);
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::getSupportedInletTypes() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_supportedInletTypesMutex);
    return m_impl->getSupportedInletTypes();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::setIsFlapOpen(bool IsFlapOpen)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    m_impl->setIsFlapOpen(IsFlapOpen);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::getIsFlapOpen() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isFlapOpenMutex);
    return m_impl->getIsFlapOpen();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    m_impl->setIsChargingCableConnected(IsChargingCableConnected);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::getIsChargingCableConnected() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableConnectedMutex);
    return m_impl->getIsChargingCableConnected();
}
void VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    m_impl->setIsChargingCableLocked(IsChargingCableLocked);
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::getIsChargingCableLocked() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingCableLockedMutex);
    return m_impl->getIsChargingCableLocked();
}

IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher& VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}