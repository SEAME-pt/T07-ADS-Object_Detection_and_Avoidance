

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargevoltage.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargevoltage.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargevoltage.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargeVoltage::VehiclePowertrainTractionBatteryChargingChargeVoltage()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargeVoltagePublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargeVoltage::~VehiclePowertrainTractionBatteryChargingChargeVoltage()
{
}

void VehiclePowertrainTractionBatteryChargingChargeVoltage::setDc(float DC)
{
    if (m_data.m_DC != DC) {
        m_data.m_DC = DC;
        m_publisher->publishDcChanged(DC);
    }
}

float VehiclePowertrainTractionBatteryChargingChargeVoltage::getDc() const
{
    return m_data.m_DC;
}

void VehiclePowertrainTractionBatteryChargingChargeVoltage::setPhase1(float Phase1)
{
    if (m_data.m_Phase1 != Phase1) {
        m_data.m_Phase1 = Phase1;
        m_publisher->publishPhase1Changed(Phase1);
    }
}

float VehiclePowertrainTractionBatteryChargingChargeVoltage::getPhase1() const
{
    return m_data.m_Phase1;
}

void VehiclePowertrainTractionBatteryChargingChargeVoltage::setPhase2(float Phase2)
{
    if (m_data.m_Phase2 != Phase2) {
        m_data.m_Phase2 = Phase2;
        m_publisher->publishPhase2Changed(Phase2);
    }
}

float VehiclePowertrainTractionBatteryChargingChargeVoltage::getPhase2() const
{
    return m_data.m_Phase2;
}

void VehiclePowertrainTractionBatteryChargingChargeVoltage::setPhase3(float Phase3)
{
    if (m_data.m_Phase3 != Phase3) {
        m_data.m_Phase3 = Phase3;
        m_publisher->publishPhase3Changed(Phase3);
    }
}

float VehiclePowertrainTractionBatteryChargingChargeVoltage::getPhase3() const
{
    return m_data.m_Phase3;
}

IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher& VehiclePowertrainTractionBatteryChargingChargeVoltage::_getPublisher() const
{
    return *m_publisher;
}
