

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportrearmiddle.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportrearmiddle.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportrearmiddle.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::~VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle()
{
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    if (m_data.m_SupportedInletTypes != SupportedInletTypes) {
        m_data.m_SupportedInletTypes = SupportedInletTypes;
        m_publisher->publishSupportedInletTypesChanged(SupportedInletTypes);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::getSupportedInletTypes() const
{
    return m_data.m_SupportedInletTypes;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::setIsFlapOpen(bool IsFlapOpen)
{
    if (m_data.m_IsFlapOpen != IsFlapOpen) {
        m_data.m_IsFlapOpen = IsFlapOpen;
        m_publisher->publishIsFlapOpenChanged(IsFlapOpen);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::getIsFlapOpen() const
{
    return m_data.m_IsFlapOpen;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    if (m_data.m_IsChargingCableConnected != IsChargingCableConnected) {
        m_data.m_IsChargingCableConnected = IsChargingCableConnected;
        m_publisher->publishIsChargingCableConnectedChanged(IsChargingCableConnected);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::getIsChargingCableConnected() const
{
    return m_data.m_IsChargingCableConnected;
}

void VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    if (m_data.m_IsChargingCableLocked != IsChargingCableLocked) {
        m_data.m_IsChargingCableLocked = IsChargingCableLocked;
        m_publisher->publishIsChargingCableLockedChanged(IsChargingCableLocked);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::getIsChargingCableLocked() const
{
    return m_data.m_IsChargingCableLocked;
}

IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher& VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle::_getPublisher() const
{
    return *m_publisher;
}
