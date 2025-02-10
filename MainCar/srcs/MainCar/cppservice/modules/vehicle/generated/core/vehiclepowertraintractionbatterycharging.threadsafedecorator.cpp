

#include "vehicle/generated/core/vehiclepowertraintractionbatterycharging.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryCharging> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setChargeLimit(int32_t ChargeLimit)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_chargeLimitMutex);
    m_impl->setChargeLimit(ChargeLimit);
}

int32_t VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getChargeLimit() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_chargeLimitMutex);
    return m_impl->getChargeLimit();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setIsCharging(bool IsCharging)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingMutex);
    m_impl->setIsCharging(IsCharging);
}

bool VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getIsCharging() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isChargingMutex);
    return m_impl->getIsCharging();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setIsDischarging(bool IsDischarging)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDischargingMutex);
    m_impl->setIsDischarging(IsDischarging);
}

bool VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getIsDischarging() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDischargingMutex);
    return m_impl->getIsDischarging();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setStartStopCharging(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_startStopChargingMutex);
    m_impl->setStartStopCharging(StartStopCharging);
}

Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getStartStopCharging() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_startStopChargingMutex);
    return m_impl->getStartStopCharging();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setAveragePower(float AveragePower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_averagePowerMutex);
    m_impl->setAveragePower(AveragePower);
}

float VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getAveragePower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_averagePowerMutex);
    return m_impl->getAveragePower();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setMaxPower(float MaxPower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxPowerMutex);
    m_impl->setMaxPower(MaxPower);
}

float VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getMaxPower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_maxPowerMutex);
    return m_impl->getMaxPower();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setEvseId(const std::string& EvseId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_evseIdMutex);
    m_impl->setEvseId(EvseId);
}

const std::string& VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getEvseId() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_evseIdMutex);
    return m_impl->getEvseId();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setPowerLoss(float PowerLoss)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossMutex);
    m_impl->setPowerLoss(PowerLoss);
}

float VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getPowerLoss() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_powerLossMutex);
    return m_impl->getPowerLoss();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setTemperature(float Temperature)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    m_impl->setTemperature(Temperature);
}

float VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getTemperature() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    return m_impl->getTemperature();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setChargeRate(float ChargeRate)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_chargeRateMutex);
    m_impl->setChargeRate(ChargeRate);
}

float VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getChargeRate() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_chargeRateMutex);
    return m_impl->getChargeRate();
}
void VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::setTimeToComplete(int32_t TimeToComplete)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeToCompleteMutex);
    m_impl->setTimeToComplete(TimeToComplete);
}

int32_t VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::getTimeToComplete() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_timeToCompleteMutex);
    return m_impl->getTimeToComplete();
}

IVehiclePowertrainTractionBatteryChargingPublisher& VehiclePowertrainTractionBatteryChargingThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}