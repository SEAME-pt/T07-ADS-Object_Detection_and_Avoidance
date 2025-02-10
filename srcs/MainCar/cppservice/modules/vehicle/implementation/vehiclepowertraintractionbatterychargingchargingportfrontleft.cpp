

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontleft.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontleft.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontleft.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::~VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft()
{
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    if (m_data.m_SupportedInletTypes != SupportedInletTypes) {
        m_data.m_SupportedInletTypes = SupportedInletTypes;
        m_publisher->publishSupportedInletTypesChanged(SupportedInletTypes);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::getSupportedInletTypes() const
{
    return m_data.m_SupportedInletTypes;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::setIsFlapOpen(bool IsFlapOpen)
{
    if (m_data.m_IsFlapOpen != IsFlapOpen) {
        m_data.m_IsFlapOpen = IsFlapOpen;
        m_publisher->publishIsFlapOpenChanged(IsFlapOpen);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::getIsFlapOpen() const
{
    return m_data.m_IsFlapOpen;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    if (m_data.m_IsChargingCableConnected != IsChargingCableConnected) {
        m_data.m_IsChargingCableConnected = IsChargingCableConnected;
        m_publisher->publishIsChargingCableConnectedChanged(IsChargingCableConnected);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::getIsChargingCableConnected() const
{
    return m_data.m_IsChargingCableConnected;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    if (m_data.m_IsChargingCableLocked != IsChargingCableLocked) {
        m_data.m_IsChargingCableLocked = IsChargingCableLocked;
        m_publisher->publishIsChargingCableLockedChanged(IsChargingCableLocked);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::getIsChargingCableLocked() const
{
    return m_data.m_IsChargingCableLocked;
}

IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher& VehiclePowertrainTractionBatteryChargingChargingPortFrontLeft::_getPublisher() const
{
    return *m_publisher;
}
