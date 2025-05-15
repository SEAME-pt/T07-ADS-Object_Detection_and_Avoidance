

#include "vehicle/generated/core/vehiclepowertraintractionbatterytemperature.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryTemperature> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::setAverage(float Average)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_averageMutex);
    m_impl->setAverage(Average);
}

float VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::getAverage() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_averageMutex);
    return m_impl->getAverage();
}
void VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::setMin(float Min)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_minMutex);
    m_impl->setMin(Min);
}

float VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::getMin() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_minMutex);
    return m_impl->getMin();
}
void VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::setMax(float Max)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxMutex);
    m_impl->setMax(Max);
}

float VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::getMax() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_maxMutex);
    return m_impl->getMax();
}
void VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::setCellTemperature(const std::list<float>& CellTemperature)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_cellTemperatureMutex);
    m_impl->setCellTemperature(CellTemperature);
}

const std::list<float>& VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::getCellTemperature() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_cellTemperatureMutex);
    return m_impl->getCellTemperature();
}

IVehiclePowertrainTractionBatteryTemperaturePublisher& VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}