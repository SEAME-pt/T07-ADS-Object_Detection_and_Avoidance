

#include "vehicle/generated/core/vehiclebodylightsbeamlow.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsBeamLowThreadSafeDecorator::VehicleBodyLightsBeamLowThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsBeamLow> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsBeamLowThreadSafeDecorator::setIsOn(bool IsOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    m_impl->setIsOn(IsOn);
}

bool VehicleBodyLightsBeamLowThreadSafeDecorator::getIsOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    return m_impl->getIsOn();
}
void VehicleBodyLightsBeamLowThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsBeamLowThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsBeamLowPublisher& VehicleBodyLightsBeamLowThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}