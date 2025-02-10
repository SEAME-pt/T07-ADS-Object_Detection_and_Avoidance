

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportrearleft.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportrearleft.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportrearleft.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::VehiclePowertrainTractionBatteryChargingChargingPortRearLeft()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortRearLeftPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::~VehiclePowertrainTractionBatteryChargingChargingPortRearLeft()
{
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    if (m_data.m_SupportedInletTypes != SupportedInletTypes) {
        m_data.m_SupportedInletTypes = SupportedInletTypes;
        m_publisher->publishSupportedInletTypesChanged(SupportedInletTypes);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::getSupportedInletTypes() const
{
    return m_data.m_SupportedInletTypes;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::setIsFlapOpen(bool IsFlapOpen)
{
    if (m_data.m_IsFlapOpen != IsFlapOpen) {
        m_data.m_IsFlapOpen = IsFlapOpen;
        m_publisher->publishIsFlapOpenChanged(IsFlapOpen);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::getIsFlapOpen() const
{
    return m_data.m_IsFlapOpen;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    if (m_data.m_IsChargingCableConnected != IsChargingCableConnected) {
        m_data.m_IsChargingCableConnected = IsChargingCableConnected;
        m_publisher->publishIsChargingCableConnectedChanged(IsChargingCableConnected);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::getIsChargingCableConnected() const
{
    return m_data.m_IsChargingCableConnected;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    if (m_data.m_IsChargingCableLocked != IsChargingCableLocked) {
        m_data.m_IsChargingCableLocked = IsChargingCableLocked;
        m_publisher->publishIsChargingCableLockedChanged(IsChargingCableLocked);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::getIsChargingCableLocked() const
{
    return m_data.m_IsChargingCableLocked;
}

IVehiclePowertrainTractionBatteryChargingChargingPortRearLeftPublisher& VehiclePowertrainTractionBatteryChargingChargingPortRearLeft::_getPublisher() const
{
    return *m_publisher;
}
