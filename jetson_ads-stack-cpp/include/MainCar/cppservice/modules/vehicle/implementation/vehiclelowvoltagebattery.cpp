

#include "vehicle/implementation/vehiclelowvoltagebattery.h"
#include "vehicle/generated/core/vehiclelowvoltagebattery.publisher.h"
#include "vehicle/generated/core/vehiclelowvoltagebattery.data.h"

using namespace Cpp::Vehicle;

VehicleLowVoltageBattery::VehicleLowVoltageBattery()
    : m_publisher(std::make_unique<VehicleLowVoltageBatteryPublisher>())
{
}
VehicleLowVoltageBattery::~VehicleLowVoltageBattery()
{
}

void VehicleLowVoltageBattery::setNominalVoltage(int32_t NominalVoltage)
{
    if (m_data.m_NominalVoltage != NominalVoltage) {
        m_data.m_NominalVoltage = NominalVoltage;
        m_publisher->publishNominalVoltageChanged(NominalVoltage);
    }
}

int32_t VehicleLowVoltageBattery::getNominalVoltage() const
{
    return m_data.m_NominalVoltage;
}

void VehicleLowVoltageBattery::setNominalCapacity(int32_t NominalCapacity)
{
    if (m_data.m_NominalCapacity != NominalCapacity) {
        m_data.m_NominalCapacity = NominalCapacity;
        m_publisher->publishNominalCapacityChanged(NominalCapacity);
    }
}

int32_t VehicleLowVoltageBattery::getNominalCapacity() const
{
    return m_data.m_NominalCapacity;
}

void VehicleLowVoltageBattery::setCurrentVoltage(float CurrentVoltage)
{
    if (m_data.m_CurrentVoltage != CurrentVoltage) {
        m_data.m_CurrentVoltage = CurrentVoltage;
        m_publisher->publishCurrentVoltageChanged(CurrentVoltage);
    }
}

float VehicleLowVoltageBattery::getCurrentVoltage() const
{
    return m_data.m_CurrentVoltage;
}

void VehicleLowVoltageBattery::setCurrentCurrent(float CurrentCurrent)
{
    if (m_data.m_CurrentCurrent != CurrentCurrent) {
        m_data.m_CurrentCurrent = CurrentCurrent;
        m_publisher->publishCurrentCurrentChanged(CurrentCurrent);
    }
}

float VehicleLowVoltageBattery::getCurrentCurrent() const
{
    return m_data.m_CurrentCurrent;
}

IVehicleLowVoltageBatteryPublisher& VehicleLowVoltageBattery::_getPublisher() const
{
    return *m_publisher;
}
