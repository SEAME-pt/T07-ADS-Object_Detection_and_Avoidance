

#include "vehicle/generated/core/vehiclebodylightsbeamhigh.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsBeamHighThreadSafeDecorator::VehicleBodyLightsBeamHighThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsBeamHigh> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsBeamHighThreadSafeDecorator::setIsOn(bool IsOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    m_impl->setIsOn(IsOn);
}

bool VehicleBodyLightsBeamHighThreadSafeDecorator::getIsOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    return m_impl->getIsOn();
}
void VehicleBodyLightsBeamHighThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsBeamHighThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsBeamHighPublisher& VehicleBodyLightsBeamHighThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}