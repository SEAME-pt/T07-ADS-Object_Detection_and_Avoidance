

#include "vehicle/generated/core/vehiclebodylightsfogfront.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsFogFrontThreadSafeDecorator::VehicleBodyLightsFogFrontThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsFogFront> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsFogFrontThreadSafeDecorator::setIsOn(bool IsOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    m_impl->setIsOn(IsOn);
}

bool VehicleBodyLightsFogFrontThreadSafeDecorator::getIsOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    return m_impl->getIsOn();
}
void VehicleBodyLightsFogFrontThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsFogFrontThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsFogFrontPublisher& VehicleBodyLightsFogFrontThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}