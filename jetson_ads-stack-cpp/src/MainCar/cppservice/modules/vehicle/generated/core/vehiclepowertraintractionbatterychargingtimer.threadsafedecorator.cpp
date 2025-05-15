

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingtimer.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingTimer> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator::setMode(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_modeMutex);
    m_impl->setMode(Mode);
}

Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator::getMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_modeMutex);
    return m_impl->getMode();
}
void VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator::setTime(const std::string& Time)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeMutex);
    m_impl->setTime(Time);
}

const std::string& VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator::getTime() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_timeMutex);
    return m_impl->getTime();
}

IVehiclePowertrainTractionBatteryChargingTimerPublisher& VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}