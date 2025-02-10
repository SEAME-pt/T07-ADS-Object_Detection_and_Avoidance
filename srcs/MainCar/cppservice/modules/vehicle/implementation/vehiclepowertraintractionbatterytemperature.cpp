

#include "vehicle/implementation/vehiclepowertraintractionbatterytemperature.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterytemperature.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterytemperature.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryTemperature::VehiclePowertrainTractionBatteryTemperature()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryTemperaturePublisher>())
{
}
VehiclePowertrainTractionBatteryTemperature::~VehiclePowertrainTractionBatteryTemperature()
{
}

void VehiclePowertrainTractionBatteryTemperature::setAverage(float Average)
{
    if (m_data.m_Average != Average) {
        m_data.m_Average = Average;
        m_publisher->publishAverageChanged(Average);
    }
}

float VehiclePowertrainTractionBatteryTemperature::getAverage() const
{
    return m_data.m_Average;
}

void VehiclePowertrainTractionBatteryTemperature::setMin(float Min)
{
    if (m_data.m_Min != Min) {
        m_data.m_Min = Min;
        m_publisher->publishMinChanged(Min);
    }
}

float VehiclePowertrainTractionBatteryTemperature::getMin() const
{
    return m_data.m_Min;
}

void VehiclePowertrainTractionBatteryTemperature::setMax(float Max)
{
    if (m_data.m_Max != Max) {
        m_data.m_Max = Max;
        m_publisher->publishMaxChanged(Max);
    }
}

float VehiclePowertrainTractionBatteryTemperature::getMax() const
{
    return m_data.m_Max;
}

void VehiclePowertrainTractionBatteryTemperature::setCellTemperature(const std::list<float>& CellTemperature)
{
    if (m_data.m_CellTemperature != CellTemperature) {
        m_data.m_CellTemperature = CellTemperature;
        m_publisher->publishCellTemperatureChanged(CellTemperature);
    }
}

const std::list<float>& VehiclePowertrainTractionBatteryTemperature::getCellTemperature() const
{
    return m_data.m_CellTemperature;
}

IVehiclePowertrainTractionBatteryTemperaturePublisher& VehiclePowertrainTractionBatteryTemperature::_getPublisher() const
{
    return *m_publisher;
}
