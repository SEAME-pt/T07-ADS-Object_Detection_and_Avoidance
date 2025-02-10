

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportanyposition.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportanyposition.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportanyposition.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::~VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition()
{
}

void VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    if (m_data.m_SupportedInletTypes != SupportedInletTypes) {
        m_data.m_SupportedInletTypes = SupportedInletTypes;
        m_publisher->publishSupportedInletTypesChanged(SupportedInletTypes);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::getSupportedInletTypes() const
{
    return m_data.m_SupportedInletTypes;
}

void VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::setIsFlapOpen(bool IsFlapOpen)
{
    if (m_data.m_IsFlapOpen != IsFlapOpen) {
        m_data.m_IsFlapOpen = IsFlapOpen;
        m_publisher->publishIsFlapOpenChanged(IsFlapOpen);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::getIsFlapOpen() const
{
    return m_data.m_IsFlapOpen;
}

void VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    if (m_data.m_IsChargingCableConnected != IsChargingCableConnected) {
        m_data.m_IsChargingCableConnected = IsChargingCableConnected;
        m_publisher->publishIsChargingCableConnectedChanged(IsChargingCableConnected);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::getIsChargingCableConnected() const
{
    return m_data.m_IsChargingCableConnected;
}

void VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    if (m_data.m_IsChargingCableLocked != IsChargingCableLocked) {
        m_data.m_IsChargingCableLocked = IsChargingCableLocked;
        m_publisher->publishIsChargingCableLockedChanged(IsChargingCableLocked);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::getIsChargingCableLocked() const
{
    return m_data.m_IsChargingCableLocked;
}

IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher& VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition::_getPublisher() const
{
    return *m_publisher;
}
