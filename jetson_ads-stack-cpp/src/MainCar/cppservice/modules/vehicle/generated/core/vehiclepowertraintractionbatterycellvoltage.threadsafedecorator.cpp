

#include "vehicle/generated/core/vehiclepowertraintractionbatterycellvoltage.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryCellVoltage> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::setMin(float Min)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_minMutex);
    m_impl->setMin(Min);
}

float VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::getMin() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_minMutex);
    return m_impl->getMin();
}
void VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::setIdMin(int32_t IdMin)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_idMinMutex);
    m_impl->setIdMin(IdMin);
}

int32_t VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::getIdMin() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_idMinMutex);
    return m_impl->getIdMin();
}
void VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::setMax(float Max)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxMutex);
    m_impl->setMax(Max);
}

float VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::getMax() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_maxMutex);
    return m_impl->getMax();
}
void VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::setIdMax(int32_t IdMax)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_idMaxMutex);
    m_impl->setIdMax(IdMax);
}

int32_t VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::getIdMax() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_idMaxMutex);
    return m_impl->getIdMax();
}
void VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::setCellVoltages(const std::list<float>& CellVoltages)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_cellVoltagesMutex);
    m_impl->setCellVoltages(CellVoltages);
}

const std::list<float>& VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::getCellVoltages() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_cellVoltagesMutex);
    return m_impl->getCellVoltages();
}

IVehiclePowertrainTractionBatteryCellVoltagePublisher& VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}