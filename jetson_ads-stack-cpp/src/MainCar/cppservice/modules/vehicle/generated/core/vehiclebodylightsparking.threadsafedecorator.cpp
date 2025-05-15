

#include "vehicle/generated/core/vehiclebodylightsparking.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsParkingThreadSafeDecorator::VehicleBodyLightsParkingThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsParking> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsParkingThreadSafeDecorator::setIsOn(bool IsOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    m_impl->setIsOn(IsOn);
}

bool VehicleBodyLightsParkingThreadSafeDecorator::getIsOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    return m_impl->getIsOn();
}
void VehicleBodyLightsParkingThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsParkingThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsParkingPublisher& VehicleBodyLightsParkingThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}