

#include "vehicle/implementation/vehiclepowertraintractionbatterycharging.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycharging.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycharging.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryCharging::VehiclePowertrainTractionBatteryCharging()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingPublisher>())
{
}
VehiclePowertrainTractionBatteryCharging::~VehiclePowertrainTractionBatteryCharging()
{
}

void VehiclePowertrainTractionBatteryCharging::setChargeLimit(int32_t ChargeLimit)
{
    if (m_data.m_ChargeLimit != ChargeLimit) {
        m_data.m_ChargeLimit = ChargeLimit;
        m_publisher->publishChargeLimitChanged(ChargeLimit);
    }
}

int32_t VehiclePowertrainTractionBatteryCharging::getChargeLimit() const
{
    return m_data.m_ChargeLimit;
}

void VehiclePowertrainTractionBatteryCharging::setIsCharging(bool IsCharging)
{
    if (m_data.m_IsCharging != IsCharging) {
        m_data.m_IsCharging = IsCharging;
        m_publisher->publishIsChargingChanged(IsCharging);
    }
}

bool VehiclePowertrainTractionBatteryCharging::getIsCharging() const
{
    return m_data.m_IsCharging;
}

void VehiclePowertrainTractionBatteryCharging::setIsDischarging(bool IsDischarging)
{
    if (m_data.m_IsDischarging != IsDischarging) {
        m_data.m_IsDischarging = IsDischarging;
        m_publisher->publishIsDischargingChanged(IsDischarging);
    }
}

bool VehiclePowertrainTractionBatteryCharging::getIsDischarging() const
{
    return m_data.m_IsDischarging;
}

void VehiclePowertrainTractionBatteryCharging::setStartStopCharging(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging)
{
    if (m_data.m_StartStopCharging != StartStopCharging) {
        m_data.m_StartStopCharging = StartStopCharging;
        m_publisher->publishStartStopChargingChanged(StartStopCharging);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum VehiclePowertrainTractionBatteryCharging::getStartStopCharging() const
{
    return m_data.m_StartStopCharging;
}

void VehiclePowertrainTractionBatteryCharging::setAveragePower(float AveragePower)
{
    if (m_data.m_AveragePower != AveragePower) {
        m_data.m_AveragePower = AveragePower;
        m_publisher->publishAveragePowerChanged(AveragePower);
    }
}

float VehiclePowertrainTractionBatteryCharging::getAveragePower() const
{
    return m_data.m_AveragePower;
}

void VehiclePowertrainTractionBatteryCharging::setMaxPower(float MaxPower)
{
    if (m_data.m_MaxPower != MaxPower) {
        m_data.m_MaxPower = MaxPower;
        m_publisher->publishMaxPowerChanged(MaxPower);
    }
}

float VehiclePowertrainTractionBatteryCharging::getMaxPower() const
{
    return m_data.m_MaxPower;
}

void VehiclePowertrainTractionBatteryCharging::setEvseId(const std::string& EvseId)
{
    if (m_data.m_EvseId != EvseId) {
        m_data.m_EvseId = EvseId;
        m_publisher->publishEvseIdChanged(EvseId);
    }
}

const std::string& VehiclePowertrainTractionBatteryCharging::getEvseId() const
{
    return m_data.m_EvseId;
}

void VehiclePowertrainTractionBatteryCharging::setPowerLoss(float PowerLoss)
{
    if (m_data.m_PowerLoss != PowerLoss) {
        m_data.m_PowerLoss = PowerLoss;
        m_publisher->publishPowerLossChanged(PowerLoss);
    }
}

float VehiclePowertrainTractionBatteryCharging::getPowerLoss() const
{
    return m_data.m_PowerLoss;
}

void VehiclePowertrainTractionBatteryCharging::setTemperature(float Temperature)
{
    if (m_data.m_Temperature != Temperature) {
        m_data.m_Temperature = Temperature;
        m_publisher->publishTemperatureChanged(Temperature);
    }
}

float VehiclePowertrainTractionBatteryCharging::getTemperature() const
{
    return m_data.m_Temperature;
}

void VehiclePowertrainTractionBatteryCharging::setChargeRate(float ChargeRate)
{
    if (m_data.m_ChargeRate != ChargeRate) {
        m_data.m_ChargeRate = ChargeRate;
        m_publisher->publishChargeRateChanged(ChargeRate);
    }
}

float VehiclePowertrainTractionBatteryCharging::getChargeRate() const
{
    return m_data.m_ChargeRate;
}

void VehiclePowertrainTractionBatteryCharging::setTimeToComplete(int32_t TimeToComplete)
{
    if (m_data.m_TimeToComplete != TimeToComplete) {
        m_data.m_TimeToComplete = TimeToComplete;
        m_publisher->publishTimeToCompleteChanged(TimeToComplete);
    }
}

int32_t VehiclePowertrainTractionBatteryCharging::getTimeToComplete() const
{
    return m_data.m_TimeToComplete;
}

IVehiclePowertrainTractionBatteryChargingPublisher& VehiclePowertrainTractionBatteryCharging::_getPublisher() const
{
    return *m_publisher;
}
