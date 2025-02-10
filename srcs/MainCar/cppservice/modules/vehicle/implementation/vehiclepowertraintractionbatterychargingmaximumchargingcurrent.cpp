

#include "vehicle/implementation/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::~VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent()
{
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::setDc(float DC)
{
    if (m_data.m_DC != DC) {
        m_data.m_DC = DC;
        m_publisher->publishDcChanged(DC);
    }
}

float VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::getDc() const
{
    return m_data.m_DC;
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::setPhase1(float Phase1)
{
    if (m_data.m_Phase1 != Phase1) {
        m_data.m_Phase1 = Phase1;
        m_publisher->publishPhase1Changed(Phase1);
    }
}

float VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::getPhase1() const
{
    return m_data.m_Phase1;
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::setPhase2(float Phase2)
{
    if (m_data.m_Phase2 != Phase2) {
        m_data.m_Phase2 = Phase2;
        m_publisher->publishPhase2Changed(Phase2);
    }
}

float VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::getPhase2() const
{
    return m_data.m_Phase2;
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::setPhase3(float Phase3)
{
    if (m_data.m_Phase3 != Phase3) {
        m_data.m_Phase3 = Phase3;
        m_publisher->publishPhase3Changed(Phase3);
    }
}

float VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::getPhase3() const
{
    return m_data.m_Phase3;
}

IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher& VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent::_getPublisher() const
{
    return *m_publisher;
}
