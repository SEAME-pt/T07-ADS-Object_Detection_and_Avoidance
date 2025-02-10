

#include "vehicle/implementation/vehiclepowertraintractionbatterystateofcharge.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterystateofcharge.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterystateofcharge.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryStateOfCharge::VehiclePowertrainTractionBatteryStateOfCharge()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryStateOfChargePublisher>())
{
}
VehiclePowertrainTractionBatteryStateOfCharge::~VehiclePowertrainTractionBatteryStateOfCharge()
{
}

void VehiclePowertrainTractionBatteryStateOfCharge::setCurrent(float Current)
{
    if (m_data.m_Current != Current) {
        m_data.m_Current = Current;
        m_publisher->publishCurrentChanged(Current);
    }
}

float VehiclePowertrainTractionBatteryStateOfCharge::getCurrent() const
{
    return m_data.m_Current;
}

void VehiclePowertrainTractionBatteryStateOfCharge::setDisplayed(float Displayed)
{
    if (m_data.m_Displayed != Displayed) {
        m_data.m_Displayed = Displayed;
        m_publisher->publishDisplayedChanged(Displayed);
    }
}

float VehiclePowertrainTractionBatteryStateOfCharge::getDisplayed() const
{
    return m_data.m_Displayed;
}

void VehiclePowertrainTractionBatteryStateOfCharge::setCurrentEnergy(float CurrentEnergy)
{
    if (m_data.m_CurrentEnergy != CurrentEnergy) {
        m_data.m_CurrentEnergy = CurrentEnergy;
        m_publisher->publishCurrentEnergyChanged(CurrentEnergy);
    }
}

float VehiclePowertrainTractionBatteryStateOfCharge::getCurrentEnergy() const
{
    return m_data.m_CurrentEnergy;
}

IVehiclePowertrainTractionBatteryStateOfChargePublisher& VehiclePowertrainTractionBatteryStateOfCharge::_getPublisher() const
{
    return *m_publisher;
}
