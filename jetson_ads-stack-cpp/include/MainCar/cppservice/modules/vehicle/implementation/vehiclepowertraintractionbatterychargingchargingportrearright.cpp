

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportrearright.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportrearright.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportrearright.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargingPortRearRight::VehiclePowertrainTractionBatteryChargingChargingPortRearRight()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargingPortRearRight::~VehiclePowertrainTractionBatteryChargingChargingPortRearRight()
{
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearRight::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    if (m_data.m_SupportedInletTypes != SupportedInletTypes) {
        m_data.m_SupportedInletTypes = SupportedInletTypes;
        m_publisher->publishSupportedInletTypesChanged(SupportedInletTypes);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortRearRight::getSupportedInletTypes() const
{
    return m_data.m_SupportedInletTypes;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearRight::setIsFlapOpen(bool IsFlapOpen)
{
    if (m_data.m_IsFlapOpen != IsFlapOpen) {
        m_data.m_IsFlapOpen = IsFlapOpen;
        m_publisher->publishIsFlapOpenChanged(IsFlapOpen);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearRight::getIsFlapOpen() const
{
    return m_data.m_IsFlapOpen;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearRight::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    if (m_data.m_IsChargingCableConnected != IsChargingCableConnected) {
        m_data.m_IsChargingCableConnected = IsChargingCableConnected;
        m_publisher->publishIsChargingCableConnectedChanged(IsChargingCableConnected);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearRight::getIsChargingCableConnected() const
{
    return m_data.m_IsChargingCableConnected;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearRight::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    if (m_data.m_IsChargingCableLocked != IsChargingCableLocked) {
        m_data.m_IsChargingCableLocked = IsChargingCableLocked;
        m_publisher->publishIsChargingCableLockedChanged(IsChargingCableLocked);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearRight::getIsChargingCableLocked() const
{
    return m_data.m_IsChargingCableLocked;
}

IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher& VehiclePowertrainTractionBatteryChargingChargingPortRearRight::_getPublisher() const
{
    return *m_publisher;
}
