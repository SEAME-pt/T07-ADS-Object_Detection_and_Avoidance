

#include "vehicle/generated/core/vehiclebodylightsrunning.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsRunningThreadSafeDecorator::VehicleBodyLightsRunningThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsRunning> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsRunningThreadSafeDecorator::setIsOn(bool IsOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    m_impl->setIsOn(IsOn);
}

bool VehicleBodyLightsRunningThreadSafeDecorator::getIsOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    return m_impl->getIsOn();
}
void VehicleBodyLightsRunningThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsRunningThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsRunningPublisher& VehicleBodyLightsRunningThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}