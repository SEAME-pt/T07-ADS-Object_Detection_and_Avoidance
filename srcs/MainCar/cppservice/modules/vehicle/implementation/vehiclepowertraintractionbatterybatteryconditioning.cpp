

#include "vehicle/implementation/vehiclepowertraintractionbatterybatteryconditioning.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterybatteryconditioning.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterybatteryconditioning.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryBatteryConditioning::VehiclePowertrainTractionBatteryBatteryConditioning()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryBatteryConditioningPublisher>())
{
}
VehiclePowertrainTractionBatteryBatteryConditioning::~VehiclePowertrainTractionBatteryBatteryConditioning()
{
}

void VehiclePowertrainTractionBatteryBatteryConditioning::setIsActive(bool IsActive)
{
    if (m_data.m_IsActive != IsActive) {
        m_data.m_IsActive = IsActive;
        m_publisher->publishIsActiveChanged(IsActive);
    }
}

bool VehiclePowertrainTractionBatteryBatteryConditioning::getIsActive() const
{
    return m_data.m_IsActive;
}

void VehiclePowertrainTractionBatteryBatteryConditioning::setIsOngoing(bool IsOngoing)
{
    if (m_data.m_IsOngoing != IsOngoing) {
        m_data.m_IsOngoing = IsOngoing;
        m_publisher->publishIsOngoingChanged(IsOngoing);
    }
}

bool VehiclePowertrainTractionBatteryBatteryConditioning::getIsOngoing() const
{
    return m_data.m_IsOngoing;
}

void VehiclePowertrainTractionBatteryBatteryConditioning::setRequestedMode(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode)
{
    if (m_data.m_RequestedMode != RequestedMode) {
        m_data.m_RequestedMode = RequestedMode;
        m_publisher->publishRequestedModeChanged(RequestedMode);
    }
}

Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum VehiclePowertrainTractionBatteryBatteryConditioning::getRequestedMode() const
{
    return m_data.m_RequestedMode;
}

void VehiclePowertrainTractionBatteryBatteryConditioning::setStartTime(const std::string& StartTime)
{
    if (m_data.m_StartTime != StartTime) {
        m_data.m_StartTime = StartTime;
        m_publisher->publishStartTimeChanged(StartTime);
    }
}

const std::string& VehiclePowertrainTractionBatteryBatteryConditioning::getStartTime() const
{
    return m_data.m_StartTime;
}

void VehiclePowertrainTractionBatteryBatteryConditioning::setTargetTime(const std::string& TargetTime)
{
    if (m_data.m_TargetTime != TargetTime) {
        m_data.m_TargetTime = TargetTime;
        m_publisher->publishTargetTimeChanged(TargetTime);
    }
}

const std::string& VehiclePowertrainTractionBatteryBatteryConditioning::getTargetTime() const
{
    return m_data.m_TargetTime;
}

void VehiclePowertrainTractionBatteryBatteryConditioning::setTargetTemperature(float TargetTemperature)
{
    if (m_data.m_TargetTemperature != TargetTemperature) {
        m_data.m_TargetTemperature = TargetTemperature;
        m_publisher->publishTargetTemperatureChanged(TargetTemperature);
    }
}

float VehiclePowertrainTractionBatteryBatteryConditioning::getTargetTemperature() const
{
    return m_data.m_TargetTemperature;
}

IVehiclePowertrainTractionBatteryBatteryConditioningPublisher& VehiclePowertrainTractionBatteryBatteryConditioning::_getPublisher() const
{
    return *m_publisher;
}
