

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingchargecurrent.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargecurrent.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargecurrent.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingChargeCurrent::VehiclePowertrainTractionBatteryChargingChargeCurrent()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingChargeCurrentPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingChargeCurrent::~VehiclePowertrainTractionBatteryChargingChargeCurrent()
{
}

void VehiclePowertrainTractionBatteryChargingChargeCurrent::setDc(float DC)
{
    if (m_data.m_DC != DC) {
        m_data.m_DC = DC;
        m_publisher->publishDcChanged(DC);
    }
}

float VehiclePowertrainTractionBatteryChargingChargeCurrent::getDc() const
{
    return m_data.m_DC;
}

void VehiclePowertrainTractionBatteryChargingChargeCurrent::setPhase1(float Phase1)
{
    if (m_data.m_Phase1 != Phase1) {
        m_data.m_Phase1 = Phase1;
        m_publisher->publishPhase1Changed(Phase1);
    }
}

float VehiclePowertrainTractionBatteryChargingChargeCurrent::getPhase1() const
{
    return m_data.m_Phase1;
}

void VehiclePowertrainTractionBatteryChargingChargeCurrent::setPhase2(float Phase2)
{
    if (m_data.m_Phase2 != Phase2) {
        m_data.m_Phase2 = Phase2;
        m_publisher->publishPhase2Changed(Phase2);
    }
}

float VehiclePowertrainTractionBatteryChargingChargeCurrent::getPhase2() const
{
    return m_data.m_Phase2;
}

void VehiclePowertrainTractionBatteryChargingChargeCurrent::setPhase3(float Phase3)
{
    if (m_data.m_Phase3 != Phase3) {
        m_data.m_Phase3 = Phase3;
        m_publisher->publishPhase3Changed(Phase3);
    }
}

float VehiclePowertrainTractionBatteryChargingChargeCurrent::getPhase3() const
{
    return m_data.m_Phase3;
}

IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher& VehiclePowertrainTractionBatteryChargingChargeCurrent::_getPublisher() const
{
    return *m_publisher;
}
