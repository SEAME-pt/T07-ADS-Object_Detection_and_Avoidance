

#include "vehicle/generated/core/vehiclebodylightsdirectionindicatorright.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator::VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsDirectionIndicatorRight> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator::setIsSignaling(bool IsSignaling)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isSignalingMutex);
    m_impl->setIsSignaling(IsSignaling);
}

bool VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator::getIsSignaling() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isSignalingMutex);
    return m_impl->getIsSignaling();
}
void VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsDirectionIndicatorRightPublisher& VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}