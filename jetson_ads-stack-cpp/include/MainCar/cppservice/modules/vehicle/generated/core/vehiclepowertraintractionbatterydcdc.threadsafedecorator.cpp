

#include "vehicle/generated/core/vehiclepowertraintractionbatterydcdc.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator::VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryDcdc> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator::setPowerLoss(float PowerLoss)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossMutex);
    m_impl->setPowerLoss(PowerLoss);
}

float VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator::getPowerLoss() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_powerLossMutex);
    return m_impl->getPowerLoss();
}
void VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator::setTemperature(float Temperature)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    m_impl->setTemperature(Temperature);
}

float VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator::getTemperature() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    return m_impl->getTemperature();
}

IVehiclePowertrainTractionBatteryDcdcPublisher& VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}