

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingtimer.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingtimer.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingtimer.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingTimer::VehiclePowertrainTractionBatteryChargingTimer()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingTimerPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingTimer::~VehiclePowertrainTractionBatteryChargingTimer()
{
}

void VehiclePowertrainTractionBatteryChargingTimer::setMode(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode)
{
    if (m_data.m_Mode != Mode) {
        m_data.m_Mode = Mode;
        m_publisher->publishModeChanged(Mode);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum VehiclePowertrainTractionBatteryChargingTimer::getMode() const
{
    return m_data.m_Mode;
}

void VehiclePowertrainTractionBatteryChargingTimer::setTime(const std::string& Time)
{
    if (m_data.m_Time != Time) {
        m_data.m_Time = Time;
        m_publisher->publishTimeChanged(Time);
    }
}

const std::string& VehiclePowertrainTractionBatteryChargingTimer::getTime() const
{
    return m_data.m_Time;
}

IVehiclePowertrainTractionBatteryChargingTimerPublisher& VehiclePowertrainTractionBatteryChargingTimer::_getPublisher() const
{
    return *m_publisher;
}
