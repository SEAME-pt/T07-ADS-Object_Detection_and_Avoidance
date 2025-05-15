

#include "vehicle/generated/core/vehiclebodylightsfogrear.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsFogRearThreadSafeDecorator::VehicleBodyLightsFogRearThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsFogRear> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsFogRearThreadSafeDecorator::setIsOn(bool IsOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    m_impl->setIsOn(IsOn);
}

bool VehicleBodyLightsFogRearThreadSafeDecorator::getIsOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    return m_impl->getIsOn();
}
void VehicleBodyLightsFogRearThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsFogRearThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsFogRearPublisher& VehicleBodyLightsFogRearThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}