

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontright.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontright.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontright.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::VehiclePowertrainTractionBatteryChargingChargingPortFrontRight()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::~VehiclePowertrainTractionBatteryChargingChargingPortFrontRight()
{
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    if (m_data.m_SupportedInletTypes != SupportedInletTypes) {
        m_data.m_SupportedInletTypes = SupportedInletTypes;
        m_publisher->publishSupportedInletTypesChanged(SupportedInletTypes);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::getSupportedInletTypes() const
{
    return m_data.m_SupportedInletTypes;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::setIsFlapOpen(bool IsFlapOpen)
{
    if (m_data.m_IsFlapOpen != IsFlapOpen) {
        m_data.m_IsFlapOpen = IsFlapOpen;
        m_publisher->publishIsFlapOpenChanged(IsFlapOpen);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::getIsFlapOpen() const
{
    return m_data.m_IsFlapOpen;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    if (m_data.m_IsChargingCableConnected != IsChargingCableConnected) {
        m_data.m_IsChargingCableConnected = IsChargingCableConnected;
        m_publisher->publishIsChargingCableConnectedChanged(IsChargingCableConnected);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::getIsChargingCableConnected() const
{
    return m_data.m_IsChargingCableConnected;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    if (m_data.m_IsChargingCableLocked != IsChargingCableLocked) {
        m_data.m_IsChargingCableLocked = IsChargingCableLocked;
        m_publisher->publishIsChargingCableLockedChanged(IsChargingCableLocked);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::getIsChargingCableLocked() const
{
    return m_data.m_IsChargingCableLocked;
}

IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher& VehiclePowertrainTractionBatteryChargingChargingPortFrontRight::_getPublisher() const
{
    return *m_publisher;
}
