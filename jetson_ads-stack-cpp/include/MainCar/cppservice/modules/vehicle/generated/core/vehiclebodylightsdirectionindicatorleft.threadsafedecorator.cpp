

#include "vehicle/generated/core/vehiclebodylightsdirectionindicatorleft.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator::VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsDirectionIndicatorLeft> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator::setIsSignaling(bool IsSignaling)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isSignalingMutex);
    m_impl->setIsSignaling(IsSignaling);
}

bool VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator::getIsSignaling() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isSignalingMutex);
    return m_impl->getIsSignaling();
}
void VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsDirectionIndicatorLeftPublisher& VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}