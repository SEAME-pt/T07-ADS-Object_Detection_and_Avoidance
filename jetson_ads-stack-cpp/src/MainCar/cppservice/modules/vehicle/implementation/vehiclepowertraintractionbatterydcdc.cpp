

#include "vehicle/implementation/vehiclepowertraintractionbatterydcdc.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterydcdc.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterydcdc.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryDcdc::VehiclePowertrainTractionBatteryDcdc()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryDcdcPublisher>())
{
}
VehiclePowertrainTractionBatteryDcdc::~VehiclePowertrainTractionBatteryDcdc()
{
}

void VehiclePowertrainTractionBatteryDcdc::setPowerLoss(float PowerLoss)
{
    if (m_data.m_PowerLoss != PowerLoss) {
        m_data.m_PowerLoss = PowerLoss;
        m_publisher->publishPowerLossChanged(PowerLoss);
    }
}

float VehiclePowertrainTractionBatteryDcdc::getPowerLoss() const
{
    return m_data.m_PowerLoss;
}

void VehiclePowertrainTractionBatteryDcdc::setTemperature(float Temperature)
{
    if (m_data.m_Temperature != Temperature) {
        m_data.m_Temperature = Temperature;
        m_publisher->publishTemperatureChanged(Temperature);
    }
}

float VehiclePowertrainTractionBatteryDcdc::getTemperature() const
{
    return m_data.m_Temperature;
}

IVehiclePowertrainTractionBatteryDcdcPublisher& VehiclePowertrainTractionBatteryDcdc::_getPublisher() const
{
    return *m_publisher;
}
