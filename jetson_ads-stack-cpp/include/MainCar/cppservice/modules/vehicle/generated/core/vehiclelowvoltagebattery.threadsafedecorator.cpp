

#include "vehicle/generated/core/vehiclelowvoltagebattery.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleLowVoltageBatteryThreadSafeDecorator::VehicleLowVoltageBatteryThreadSafeDecorator(std::shared_ptr<IVehicleLowVoltageBattery> impl)
    : m_impl(impl)
{
}
void VehicleLowVoltageBatteryThreadSafeDecorator::setNominalVoltage(int32_t NominalVoltage)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalVoltageMutex);
    m_impl->setNominalVoltage(NominalVoltage);
}

int32_t VehicleLowVoltageBatteryThreadSafeDecorator::getNominalVoltage() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_nominalVoltageMutex);
    return m_impl->getNominalVoltage();
}
void VehicleLowVoltageBatteryThreadSafeDecorator::setNominalCapacity(int32_t NominalCapacity)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalCapacityMutex);
    m_impl->setNominalCapacity(NominalCapacity);
}

int32_t VehicleLowVoltageBatteryThreadSafeDecorator::getNominalCapacity() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_nominalCapacityMutex);
    return m_impl->getNominalCapacity();
}
void VehicleLowVoltageBatteryThreadSafeDecorator::setCurrentVoltage(float CurrentVoltage)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentVoltageMutex);
    m_impl->setCurrentVoltage(CurrentVoltage);
}

float VehicleLowVoltageBatteryThreadSafeDecorator::getCurrentVoltage() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_currentVoltageMutex);
    return m_impl->getCurrentVoltage();
}
void VehicleLowVoltageBatteryThreadSafeDecorator::setCurrentCurrent(float CurrentCurrent)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentCurrentMutex);
    m_impl->setCurrentCurrent(CurrentCurrent);
}

float VehicleLowVoltageBatteryThreadSafeDecorator::getCurrentCurrent() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_currentCurrentMutex);
    return m_impl->getCurrentCurrent();
}

IVehicleLowVoltageBatteryPublisher& VehicleLowVoltageBatteryThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}