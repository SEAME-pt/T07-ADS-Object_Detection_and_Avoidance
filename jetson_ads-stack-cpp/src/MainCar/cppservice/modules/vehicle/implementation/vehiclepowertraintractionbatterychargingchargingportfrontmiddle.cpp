

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargingportfrontmiddle.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontmiddle.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontmiddle.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::~VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle()
{
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum SupportedInletTypes)
{
    if (m_data.m_SupportedInletTypes != SupportedInletTypes) {
        m_data.m_SupportedInletTypes = SupportedInletTypes;
        m_publisher->publishSupportedInletTypesChanged(SupportedInletTypes);
    }
}

Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::getSupportedInletTypes() const
{
    return m_data.m_SupportedInletTypes;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::setIsFlapOpen(bool IsFlapOpen)
{
    if (m_data.m_IsFlapOpen != IsFlapOpen) {
        m_data.m_IsFlapOpen = IsFlapOpen;
        m_publisher->publishIsFlapOpenChanged(IsFlapOpen);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::getIsFlapOpen() const
{
    return m_data.m_IsFlapOpen;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::setIsChargingCableConnected(bool IsChargingCableConnected)
{
    if (m_data.m_IsChargingCableConnected != IsChargingCableConnected) {
        m_data.m_IsChargingCableConnected = IsChargingCableConnected;
        m_publisher->publishIsChargingCableConnectedChanged(IsChargingCableConnected);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::getIsChargingCableConnected() const
{
    return m_data.m_IsChargingCableConnected;
}

void VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::setIsChargingCableLocked(bool IsChargingCableLocked)
{
    if (m_data.m_IsChargingCableLocked != IsChargingCableLocked) {
        m_data.m_IsChargingCableLocked = IsChargingCableLocked;
        m_publisher->publishIsChargingCableLockedChanged(IsChargingCableLocked);
    }
}

bool VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::getIsChargingCableLocked() const
{
    return m_data.m_IsChargingCableLocked;
}

IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher& VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle::_getPublisher() const
{
    return *m_publisher;
}
