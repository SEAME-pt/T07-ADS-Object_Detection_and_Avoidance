

#include "vehicle/generated/core/vehiclebodylightslicenseplate.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsLicensePlateThreadSafeDecorator::VehicleBodyLightsLicensePlateThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsLicensePlate> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsLicensePlateThreadSafeDecorator::setIsOn(bool IsOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    m_impl->setIsOn(IsOn);
}

bool VehicleBodyLightsLicensePlateThreadSafeDecorator::getIsOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    return m_impl->getIsOn();
}
void VehicleBodyLightsLicensePlateThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsLicensePlateThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsLicensePlatePublisher& VehicleBodyLightsLicensePlateThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}