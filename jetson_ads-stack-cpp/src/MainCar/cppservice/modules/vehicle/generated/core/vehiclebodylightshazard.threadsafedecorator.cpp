

#include "vehicle/generated/core/vehiclebodylightshazard.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsHazardThreadSafeDecorator::VehicleBodyLightsHazardThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsHazard> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsHazardThreadSafeDecorator::setIsSignaling(bool IsSignaling)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isSignalingMutex);
    m_impl->setIsSignaling(IsSignaling);
}

bool VehicleBodyLightsHazardThreadSafeDecorator::getIsSignaling() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isSignalingMutex);
    return m_impl->getIsSignaling();
}
void VehicleBodyLightsHazardThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsHazardThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsHazardPublisher& VehicleBodyLightsHazardThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}