

#include "vehicle/implementation/vehiclepowertraintractionbatterycellvoltage.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycellvoltage.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycellvoltage.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryCellVoltage::VehiclePowertrainTractionBatteryCellVoltage()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryCellVoltagePublisher>())
{
}
VehiclePowertrainTractionBatteryCellVoltage::~VehiclePowertrainTractionBatteryCellVoltage()
{
}

void VehiclePowertrainTractionBatteryCellVoltage::setMin(float Min)
{
    if (m_data.m_Min != Min) {
        m_data.m_Min = Min;
        m_publisher->publishMinChanged(Min);
    }
}

float VehiclePowertrainTractionBatteryCellVoltage::getMin() const
{
    return m_data.m_Min;
}

void VehiclePowertrainTractionBatteryCellVoltage::setIdMin(int32_t IdMin)
{
    if (m_data.m_IdMin != IdMin) {
        m_data.m_IdMin = IdMin;
        m_publisher->publishIdMinChanged(IdMin);
    }
}

int32_t VehiclePowertrainTractionBatteryCellVoltage::getIdMin() const
{
    return m_data.m_IdMin;
}

void VehiclePowertrainTractionBatteryCellVoltage::setMax(float Max)
{
    if (m_data.m_Max != Max) {
        m_data.m_Max = Max;
        m_publisher->publishMaxChanged(Max);
    }
}

float VehiclePowertrainTractionBatteryCellVoltage::getMax() const
{
    return m_data.m_Max;
}

void VehiclePowertrainTractionBatteryCellVoltage::setIdMax(int32_t IdMax)
{
    if (m_data.m_IdMax != IdMax) {
        m_data.m_IdMax = IdMax;
        m_publisher->publishIdMaxChanged(IdMax);
    }
}

int32_t VehiclePowertrainTractionBatteryCellVoltage::getIdMax() const
{
    return m_data.m_IdMax;
}

void VehiclePowertrainTractionBatteryCellVoltage::setCellVoltages(const std::list<float>& CellVoltages)
{
    if (m_data.m_CellVoltages != CellVoltages) {
        m_data.m_CellVoltages = CellVoltages;
        m_publisher->publishCellVoltagesChanged(CellVoltages);
    }
}

const std::list<float>& VehiclePowertrainTractionBatteryCellVoltage::getCellVoltages() const
{
    return m_data.m_CellVoltages;
}

IVehiclePowertrainTractionBatteryCellVoltagePublisher& VehiclePowertrainTractionBatteryCellVoltage::_getPublisher() const
{
    return *m_publisher;
}
